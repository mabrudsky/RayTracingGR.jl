using RayTracingGR
import Test: @testset, @test, @test_throws
import StaticArrays: SVector
import KernelAbstractions
import SciMLBase
import OrdinaryDiffEqTsit5: Tsit5
import DiffEqGPU: GPUTsit5, EnsembleGPUKernel

const RTC = RayTracingGR

@testset "PulsarRegion - constructors" begin
    @test PulsarRegion(5.0, 50.0)            isa PulsarRegion{Float64}
    @test PulsarRegion(R_in=5.0, R_out=50.0) isa PulsarRegion{Float64}
    @test PulsarRegion(5f0, 50f0)            isa PulsarRegion{Float32}
    @test isbitstype(PulsarRegion{Float64})

    @test_throws ArgumentError PulsarRegion(5, 50.0)     # Int is not allowed
    @test_throws ArgumentError PulsarRegion(5.0, 50f0)   # mixed precision
    @test_throws ArgumentError PulsarRegion(0.0, 50.0)   # R_in must be positive
    @test_throws ArgumentError PulsarRegion(50.0, 5.0)   # R_out > R_in
end

@testset "PulsarRegion - condition in each coordinate system" begin
    region = PulsarRegion(5.0, 50.0)
    ks = Kerr.KerrSchildCoordinates(1.0, 0.1)
    bl = Kerr.BoyerLindquistCoordinates(1.0, 0.1)
    # Boyer-Lindquist (t, r, θ, φ): inside, on the borders and outside
    @test RTC.region_condition(region, bl, SVector(0.0, 10.0, 1.0, 0.5)) > 0
    @test RTC.region_condition(region, bl, SVector(0.0,  5.0, 1.0, 0.5)) == 0
    @test RTC.region_condition(region, bl, SVector(0.0, 50.0, 1.0, 0.5)) == 0
    @test RTC.region_condition(region, bl, SVector(0.0,  4.0, 1.0, 0.5)) < 0
    @test RTC.region_condition(region, bl, SVector(0.0, 60.0, 1.0, 0.5)) < 0
    # Kerr-Schild (t, x, y, z): on the axis r = |z|
    @test RTC.region_condition(region, ks, SVector(0.0, 0.0, 0.0, 10.0)) > 0
    @test RTC.region_condition(region, ks, SVector(0.0, 0.0, 0.0,  4.0)) < 0
    @test RTC.region_condition(region, ks, SVector(0.0, 0.0, 0.0, 60.0)) < 0
end

# Photons emitted between r = 8 and r = 15 in random-like directions, in Kerr-Schild coordinates
function pulsar_rays(spacetime, n, ::Type{T}) where T
    data = zeros(T, 8, n)
    for j in 1:n
        θ, φ = T(0.3 + 2.5 * j / n), T(2π * j * 0.618)                  # positions spread over the sphere
        rad  = T(8 + 7 * ((j * 0.371) % 1))
        x    = SVector{4,T}(0, rad * sin(θ) * cos(φ), rad * sin(θ) * sin(φ), rad * cos(θ))
        α, β = T(π * ((j * 0.733) % 1)), T(2π * ((j * 0.419) % 1))       # direction of emission
        n3   = SVector{3,T}(sin(α) * cos(β), sin(α) * sin(β), cos(α))
        g    = metric(spacetime, x); ks = SVector(zero(T), n3...)        # k^t from g_{μν} k^μ k^ν = 0
        a = g[1,1]; b = 2 * (g[1,2] * n3[1] + g[1,3] * n3[2] + g[1,4] * n3[3]); c = ks' * g * ks
        data[:, j] = vcat(x, SVector((-b - sqrt(b^2 - 4a*c)) / (2a), n3...))
    end
    return data
end

@testset "boundary_callback - rays stop at the borders ($T)" for T in (Float64, Float32)
    spacetime   = Kerr.KerrSchildCoordinates(T(1), T(0.1))
    region = PulsarRegion(T(5), T(50))
    n    = 40
    data = pulsar_rays(spacetime, n, T)
    cfg  = RTC.Configuration(spacetime = spacetime, equations = RTC.GeodesicEquationsVacuum(),
                             backend = KernelAbstractions.CPU(), u0 = SVector{8,T}(data[:, 1]),
                             tspan = (zero(T), T(500)))
    ep   = RTC.Ensemble_Problem(cfg, data)

    # The type of callback is chosen from the backend
    cb_cpu = boundary_callback(region, cfg)                 # configuration.backend = CPU()
    cb_gpu = boundary_callback(region, FakeGPUBackend())    # any KernelAbstractions.GPU (defined in ensemble_tests.jl)
    @test cb_cpu isa SciMLBase.ContinuousCallback
    @test cb_gpu isa SciMLBase.DiscreteCallback
    @test boundary_callback(region, KernelAbstractions.CPU()) isa SciMLBase.ContinuousCallback
    @test isbits(cb_gpu.condition) && isbits(cb_gpu.affect!)   # they travel to the GPU kernel

    tol  = T == Float64 ? T(1e-10) : T(1e-6)
    atol_border = T == Float64 ? 1e-10 : 1e-3

    # CPU: Tsit5 + ContinuousCallback; GPU kernel (run on the CPU): GPUTsit5 + DiscreteCallback with bisection
    cpu = SciMLBase.solve(ep, Tsit5(), SciMLBase.EnsembleThreads(); trajectories = n, callback = cb_cpu,
                          save_everystep = false, save_start = false, abstol = tol, reltol = tol)
    gpu = SciMLBase.solve(ep, GPUTsit5(), EnsembleGPUKernel(KernelAbstractions.CPU()); trajectories = n,
                          callback = cb_gpu, merge_callbacks = true, adaptive = true,
                          save_everystep = false, save_start = false, abstol = tol, reltol = tol)

    r_cpu = [coordinate_radius(spacetime, SVector(u[1], u[2], u[3], u[4])) for u in cpu.u]
    r_gpu = [coordinate_radius(spacetime, SVector(u[1], u[2], u[3], u[4])) for u in gpu.u]
    on_border(r) = abs(r - region.R_in) < atol_border || abs(r - region.R_out) < atol_border

    @test all(on_border, r_cpu)                     # every ray stopped at R_in or R_out (CPU)
    @test all(on_border, r_gpu)                     # every ray stopped at R_in or R_out (GPU kernel)
    @test count(r -> r < 10, r_cpu) == count(r -> r < 10, r_gpu)   # same rays hit the star
    @test any(r -> r < 10, r_cpu) && any(r -> r > 10, r_cpu)       # both borders are tested
    rel = maximum(maximum(abs.(cpu.u[i] .- gpu.u[i]) ./ max.(1, abs.(cpu.u[i]))) for i in 1:n)
    @test rel < (T == Float64 ? 1e-7 : 1e-3)       # CPU and GPU kernel give the same final states
end
