using RayTracingGR
import Test: @testset, @test, @test_throws
import StaticArrays: SVector, @SVector
import KernelAbstractions
import SciMLBase

const RT = RayTracingGR

# Allocations are measured inside a function: measured at global scope they give false positives.
allocated_rhs(f, u, p) = (f(u, p, zero(eltype(u))); @allocated f(u, p, zero(eltype(u))))

# Array that reports a GPU backend, to check that Ensemble_Problem rejects GPU arrays without a GPU
struct FakeGPUBackend <: KernelAbstractions.GPU end
struct FakeGPUArray{T} <: AbstractMatrix{T}
    data::Matrix{T}
end
Base.size(A::FakeGPUArray) = size(A.data)
Base.getindex(A::FakeGPUArray, i::Int, j::Int) = A.data[i, j]
KernelAbstractions.get_backend(::FakeGPUArray) = FakeGPUBackend()

configuration(spacetime, equations, u0) =
    RT.Configuration(spacetime = spacetime, equations = equations, backend = KernelAbstractions.CPU(),
                     u0 = u0, tspan = (zero(eltype(u0)), one(eltype(u0))))

@testset "Ensemble_Problem - valid configurations" begin
    st64 = Kerr.KerrSchildCoordinates(1.0, 0.5)
    st32 = Kerr.KerrSchildCoordinates(1f0, 0.5f0)

    cfg = configuration(st64, RT.GeodesicEquationsVacuum(), @SVector(zeros(8)))
    ep  = RT.Ensemble_Problem(cfg, zeros(8, 3))
    @test ep isa SciMLBase.EnsembleProblem
    @test ep.prob.f.f === RT.geodesic_equations_Vacuum

    cfg = configuration(st32, RT.ParallelTransportEquationsVacuum(), @SVector(zeros(Float32, 16)))
    ep  = RT.Ensemble_Problem(cfg, zeros(Float32, 16, 3))
    @test ep isa SciMLBase.EnsembleProblem
    @test ep.prob.f.f === RT.parallel_transport_equations_Vacuum

    # Other CPU arrays are accepted too
    cfg = configuration(st64, RT.GeodesicEquationsVacuum(), @SVector(zeros(8)))
    @test RT.Ensemble_Problem(cfg, view(zeros(8, 5), :, 1:3)) isa SciMLBase.EnsembleProblem
end

@testset "Ensemble_Problem - invalid configurations" begin
    st64 = Kerr.KerrSchildCoordinates(1.0, 0.5)
    st32 = Kerr.KerrSchildCoordinates(1f0, 0.5f0)
    geodesic = RT.GeodesicEquationsVacuum()
    parallel = RT.ParallelTransportEquationsVacuum()

    # u0 length does not match the equation set
    @test_throws DimensionMismatch RT.Ensemble_Problem(configuration(st64, parallel, @SVector(zeros(8))), zeros(8, 3))
    @test_throws DimensionMismatch RT.Ensemble_Problem(configuration(st64, geodesic, @SVector(zeros(16))), zeros(16, 3))
    # spacetime and u0 with different precision
    @test_throws ArgumentError RT.Ensemble_Problem(configuration(st64, geodesic, @SVector(zeros(Float32, 8))), zeros(Float32, 8, 3))
    @test_throws ArgumentError RT.Ensemble_Problem(configuration(st32, geodesic, @SVector(zeros(8))), zeros(8, 3))
    # initial_data rows do not match u0
    @test_throws DimensionMismatch RT.Ensemble_Problem(configuration(st64, geodesic, @SVector(zeros(8))), zeros(7, 3))
    # initial_data element type does not match u0
    @test_throws ArgumentError RT.Ensemble_Problem(configuration(st64, geodesic, @SVector(zeros(8))), zeros(Float32, 8, 3))
    # initial_data on the GPU
    @test_throws ArgumentError RT.Ensemble_Problem(configuration(st64, geodesic, @SVector(zeros(8))), FakeGPUArray(zeros(8, 3)))
    # a coordinate system is not a spacetime
    @test_throws MethodError configuration(RT.CartesianCoordinate(), geodesic, @SVector(zeros(8)))
end

@testset "Equations - isbits and zero allocations ($T)" for T in (Float64, Float32)
    spacetime = Kerr.KerrSchildCoordinates(T(1), T(0.9))
    p   = RT.IntegrationParameters(spacetime)
    u8  = SVector{8,T}(0.3, 3.0, 0.4, 1.2,  1.0, 0.2, -0.1, 0.05)
    u16 = SVector{16,T}(0.3, 3.0, 0.4, 1.2,  1.0, 0.2, -0.1, 0.05,  0.0, 1.0, 0.3, 0.0,  0.0, 0.1, 1.0, 0.2)

    @test isbitstype(typeof(p))
    @test RT.geodesic_equations_Vacuum(u8, p, zero(T)) isa SVector{8,T}
    @test RT.parallel_transport_equations_Vacuum(u16, p, zero(T)) isa SVector{16,T}
    @test allocated_rhs(RT.geodesic_equations_Vacuum, u8, p) == 0
    @test allocated_rhs(RT.parallel_transport_equations_Vacuum, u16, p) == 0
end
