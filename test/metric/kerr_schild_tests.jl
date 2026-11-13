using RayTracingGR
import Test: @testset, @test, @test_throws
import StaticArrays: SVector

# Allocations are measured inside a function: measured at global scope they give false positives.
allocated_metric(st, x)         = (metric(st, x);         @allocated metric(st, x))
allocated_metric_inverse(st, x) = (metric_inverse(st, x); @allocated metric_inverse(st, x))
allocated_christoffel(st, x)    = (christoffel(st, x);    @allocated christoffel(st, x))

@testset "KerrSchildCoordinates - constructors" begin
    @test Kerr.KerrSchildCoordinates(1.0, 0.5)       isa Kerr.KerrSchildCoordinates{Float64}
    @test Kerr.KerrSchildCoordinates(M=1.0, a=0.5)   isa Kerr.KerrSchildCoordinates{Float64}
    @test Kerr.KerrSchildCoordinates(1f0, 0.5f0)     isa Kerr.KerrSchildCoordinates{Float32}
    @test Kerr.KerrSchildCoordinates(1.0, 0.5) == Kerr.KerrSchildCoordinates(M=1.0, a=0.5)

    @test_throws ArgumentError Kerr.KerrSchildCoordinates(1, 0.5)      # Int is not allowed
    @test_throws ArgumentError Kerr.KerrSchildCoordinates(1.0, 0.5f0)  # mixed precision
    @test_throws ArgumentError Kerr.KerrSchildCoordinates(-1.0, 0.0)   # M < 0
    @test_throws ArgumentError Kerr.KerrSchildCoordinates(1.0, 2.0)    # |a| > M
end

@testset "KerrSchildCoordinates - isbits and zero allocations ($T)" for T in (Float64, Float32)
    spacetime = Kerr.KerrSchildCoordinates(T(1), T(0.9))
    x = SVector{4,T}(0.3, 3.0, 0.4, 1.2)

    @test isbitstype(typeof(spacetime))
    @test isbitstype(typeof(metric(spacetime, x)))
    @test isbitstype(typeof(metric_inverse(spacetime, x)))
    @test isbitstype(typeof(christoffel(spacetime, x)))
    @test eltype(christoffel(spacetime, x)) === T

    @test allocated_metric(spacetime, x) == 0
    @test allocated_metric_inverse(spacetime, x) == 0
    @test allocated_christoffel(spacetime, x) == 0
end
