using RayTracingGR
import Test: @testset, @test, @test_throws
import StaticArrays: SVector
import KernelAbstractions
import SciMLBase
import OrdinaryDiffEqTsit5: Tsit5
import OrdinaryDiffEqVerner: Vern9
import DiffEqGPU: GPUTsit5

const RTS = RayTracingGR
# Uses pulsar_rays (callback_tests.jl) and FakeGPUBackend (ensemble_tests.jl)

pulsar_configuration(spacetime, data, backend) =
    RTS.Configuration(spacetime = spacetime, equations = RTS.GeodesicEquationsVacuum(), backend = backend,
                      u0 = SVector{8}(data[:, 1]), tspan = (zero(eltype(data)), eltype(data)(500)))

@testset "propagate - defaults" begin
    @test RTS.default_algorithm(KernelAbstractions.CPU()) isa Vern9
    @test RTS.default_algorithm(FakeGPUBackend())         isa GPUTsit5
    @test RTS.default_tolerance(Float64) == 5 * eps(Float64)
    @test RTS.default_tolerance(Float32) == 5 * eps(Float32)
end

# Distance of each ray to the nearest border, and null condition |g_{μν} k^μ k^ν| / (k^t)^2 at the end
distance_to_border(spacetime, region, final) =
    [begin r = coordinate_radius(spacetime, SVector{4}(final[1:4, j])); min(abs(r - region.R_in), abs(r - region.R_out)) end for j in axes(final, 2)]
null_condition(spacetime, final) =
    [begin g = metric(spacetime, SVector{4}(final[1:4, j])); k = SVector{4}(final[5:8, j]); abs(k' * g * k) / k[1]^2 end for j in axes(final, 2)]

# Each case with the precision it is meant for: Float64 on the CPU (Vern9 by default),
# Float32 on the GPU (GPUTsit5 by default; here Tsit5, its CPU version, since the CI has no GPU)
@testset "propagate - CPU, Float64, default algorithm (Vern9)" begin
    spacetime = Kerr.KerrSchildCoordinates(1.0, 0.1)
    region    = PulsarRegion(5.0, 50.0)
    n         = 40
    data      = pulsar_rays(spacetime, n, Float64)
    cfg       = pulsar_configuration(spacetime, data, KernelAbstractions.CPU())

    final = propagate(cfg, data, region)                       # Vern9, tolerance 5 eps(Float64)
    @test final isa Matrix{Float64}
    @test size(final) == (8, n)                                # one column per ray, like initial_data
    @test maximum(distance_to_border(spacetime, region, final)) < 1e-10
    @test maximum(null_condition(spacetime, final)) < 1e-12

    # The algorithm and the options of solve can be changed
    final_tsit5 = propagate(cfg, data, region; alg = Tsit5(), reltol = 1e-8, abstol = 1e-8)
    @test size(final_tsit5) == (8, n)
    @test maximum(abs.(final_tsit5 .- final) ./ max.(abs.(final), 1)) < 1e-5

    # Same call with every argument given by name, like Configuration
    @test propagate(configuration = cfg, initial_data = data, region = region) == final
    @test propagate(configuration = cfg, initial_data = data, region = region,
                    alg = Tsit5(), reltol = 1e-8, abstol = 1e-8) == final_tsit5
end

@testset "propagate - Float32 (Tsit5)" begin
    spacetime = Kerr.KerrSchildCoordinates(1f0, 0.1f0)
    region    = PulsarRegion(5f0, 50f0)
    n         = 40
    data      = pulsar_rays(spacetime, n, Float32)
    cfg       = pulsar_configuration(spacetime, data, KernelAbstractions.CPU())

    final = propagate(cfg, data, region; alg = Tsit5())        # tolerance 5 eps(Float32)
    @test final isa Matrix{Float32}
    @test size(final) == (8, n)
    @test maximum(distance_to_border(spacetime, region, final)) < 1e-4
    @test maximum(null_condition(spacetime, final)) < 1e-4
end

@testset "propagate - invalid inputs" begin
    spacetime = Kerr.KerrSchildCoordinates(1.0, 0.1)
    region    = PulsarRegion(5.0, 50.0)
    data      = pulsar_rays(spacetime, 4, Float64)
    cpu       = pulsar_configuration(spacetime, data, KernelAbstractions.CPU())
    gpu       = pulsar_configuration(spacetime, data, FakeGPUBackend())

    @test_throws ArgumentError propagate(cpu, data, region; alg = GPUTsit5())        # GPU algorithm on the CPU
    @test_throws ArgumentError propagate(gpu, data, region; alg = Tsit5())           # CPU algorithm on a GPU
    @test_throws ArgumentError propagate(cpu, data, region; callback = nothing)      # set by propagate
    @test_throws ArgumentError propagate(cpu, data, region; trajectories = 2)
    @test_throws ArgumentError propagate(cpu, data, region; batch_size = 10)
    @test_throws ArgumentError propagate(cpu, data, region; merge_callbacks = false)
end
