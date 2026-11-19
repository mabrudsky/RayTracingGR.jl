using RayTracingGR
import Test: @testset, @test, @test_throws
import StaticArrays: SVector
import KernelAbstractions
import SciMLBase
import OrdinaryDiffEqTsit5: Tsit5
import DiffEqGPU: GPUTsit5, EnsembleGPUKernel

const RTD = RayTracingGR
# Uses FakeGPUBackend (ensemble_tests.jl)

@testset "AccretionDiskRegion - constructors" begin
    @test AccretionDiskRegion(2.0, 60.0, 4.0, 15.0)                             isa AccretionDiskRegion{Float64}
    @test AccretionDiskRegion(R_min = 2.0, R_max = 60.0, R_in = 4.0, R_out = 15.0) isa AccretionDiskRegion{Float64}
    @test AccretionDiskRegion(2f0, 60f0, 4f0, 15f0)                             isa AccretionDiskRegion{Float32}
    @test AccretionDiskRegion(2.0, 60.0, 2.0, 15.0)                             isa AccretionDiskRegion{Float64}   # R_in = R_min (as in Skylight)
    @test isbitstype(AccretionDiskRegion{Float64})

    @test_throws ArgumentError AccretionDiskRegion(2, 60.0, 4.0, 15.0)      # Int is not allowed
    @test_throws ArgumentError AccretionDiskRegion(2.0, 60f0, 4.0, 15.0)    # mixed precision
    @test_throws ArgumentError AccretionDiskRegion(-1.0, 60.0, 4.0, 15.0)   # R_min ≥ 0
    @test_throws ArgumentError AccretionDiskRegion(5.0, 60.0, 4.0, 15.0)    # R_in ≥ R_min
    @test_throws ArgumentError AccretionDiskRegion(2.0, 60.0, 15.0, 4.0)    # R_out ≥ R_in
    @test_throws ArgumentError AccretionDiskRegion(2.0, 10.0, 4.0, 15.0)    # R_max > R_out
end

@testset "coordinate_z and the disk condition in each coordinate system" begin
    ks = Kerr.KerrSchildCoordinates(1.0, 0.5)
    bl = Kerr.BoyerLindquistCoordinates(1.0, 0.5)
    @test coordinate_z(ks, SVector(0.0, 1.0, 2.0, 3.0)) == 3.0
    @test coordinate_z(bl, SVector(0.0, 10.0, π / 3, 0.5)) ≈ 10.0 * cos(π / 3)
    @test abs(coordinate_z(bl, SVector(0.0, 10.0, π / 2, 0.5))) < 1e-14       # equatorial plane

    disk = AccretionDiskRegion(2.0, 60.0, 4.0, 15.0)
    # Boyer-Lindquist (t, r, θ, φ): the condition changes sign across the plane and at R_min, R_max
    @test RTD.region_condition(disk, bl, SVector(0.0, 10.0, 1.0, 0.0)) > 0     # above the plane, between the radii
    @test RTD.region_condition(disk, bl, SVector(0.0, 10.0, 2.0, 0.0)) < 0     # below the plane
    @test RTD.region_condition(disk, bl, SVector(0.0, 70.0, 1.0, 0.0)) < 0     # beyond R_max
    # Which crossings stop the ray
    @test  RTD.stops_at_boundary(disk, bl, SVector(0.0, 10.0, π / 2, 0.0))     # plane inside the disk
    @test !RTD.stops_at_boundary(disk, bl, SVector(0.0, 30.0, π / 2, 0.0))     # plane outside the disk
    @test  RTD.stops_at_boundary(disk, bl, SVector(0.0, 2.0, 1.0, 0.0))        # r = R_min
    @test  RTD.stops_at_boundary(disk, bl, SVector(0.0, 60.0, 1.0, 0.0))       # r = R_max
    # PulsarRegion: every crossing stops (default)
    @test  RTD.stops_at_boundary(PulsarRegion(5.0, 50.0), bl, SVector(0.0, 5.0, 1.0, 0.0))

    # The GPU version of the callback travels to the GPU kernel: condition and affect! must be isbits
    cb_gpu = boundary_callback(disk, FakeGPUBackend())
    @test cb_gpu isa SciMLBase.DiscreteCallback
    @test isbits(cb_gpu.condition) && isbits(cb_gpu.affect!)
    @test boundary_callback(disk, KernelAbstractions.CPU()) isa SciMLBase.ContinuousCallback
end

# Ray from the point x (Kerr-Schild) in the spatial direction n, with null momentum (g_{μν} k^μ k^ν = 0)
function disk_ray(spacetime, x::SVector{4,T}, n::SVector{3,T}) where T
    g = metric(spacetime, x); ks = SVector(zero(T), n...)
    a = g[1,1]; b = 2 * (g[1,2] * n[1] + g[1,3] * n[2] + g[1,4] * n[3]); c = ks' * g * ks
    return vcat(x, SVector((-b - sqrt(b^2 - 4a*c)) / (2a), n...))
end

@testset "AccretionDiskRegion - rays hit the disk, cross the plane outside it, fall or escape ($T)" for T in (Float64, Float32)
    spacetime = Kerr.KerrSchildCoordinates(T(1), T(0.5))
    disk      = AccretionDiskRegion(T(2), T(60), T(4), T(15))
    down, up  = SVector{3,T}(0, 0, -1), SVector{3,T}(0, 0, 1)
    rays = [disk_ray(spacetime, SVector{4,T}(0, 8, 0, 10),   down),   # A: hits the disk
            disk_ray(spacetime, SVector{4,T}(0, 30, 0, 10),  down),   # B: crosses the plane outside the disk, escapes below
            disk_ray(spacetime, SVector{4,T}(0, 0.3, 0, 10), down),   # C: falls to R_min
            disk_ray(spacetime, SVector{4,T}(0, 8, 0, 10),   up)]     # D: escapes above
    data = reduce(hcat, rays)

    cfg = RTD.Configuration(spacetime = spacetime, equations = RTD.GeodesicEquationsVacuum(),
                            backend = KernelAbstractions.CPU(), u0 = SVector{8}(data[:, 1]), tspan = (zero(T), T(500)))
    tol = T == Float64 ? T(1e-10) : T(1e-6)
    ep  = RTD.Ensemble_Problem(cfg, data)

    # CPU: ContinuousCallback; GPU kernel (run on the CPU): DiscreteCallback with bisection + Newton step
    cpu = propagate(cfg, data, disk; alg = Tsit5(), abstol = tol, reltol = tol)
    gpu = stack(SciMLBase.solve(ep, GPUTsit5(), EnsembleGPUKernel(KernelAbstractions.CPU()); trajectories = 4,
                                callback = boundary_callback(disk, FakeGPUBackend()), merge_callbacks = true,
                                adaptive = true, save_everystep = false, save_start = false, abstol = tol, reltol = tol).u)

    at = T == Float64 ? 1e-8 : 1e-3
    for (name, final) in (("CPU", cpu), ("GPU kernel", gpu))
        r(j) = coordinate_radius(spacetime, SVector{4}(final[1:4, j]))
        z(j) = coordinate_z(spacetime, SVector{4}(final[1:4, j]))
        @test abs(z(1)) < at && disk.R_in <= r(1) <= disk.R_out     # A: on the disk
        @test abs(r(2) - disk.R_max) < at && z(2) < 0                # B: went through the plane (outside the disk) and escaped below
        @test abs(r(3) - disk.R_min) < at                            # C: fell to R_min
        @test abs(r(4) - disk.R_max) < at && z(4) > 0                # D: escaped above
    end
    @test maximum(abs.(cpu .- gpu) ./ max.(abs.(cpu), 1)) < (T == Float64 ? 1e-6 : 1e-2)   # same final states
end
