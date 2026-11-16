using RayTracingGR
import Test: @testset, @test, @test_throws
import StaticArrays: SVector

# Allocations are measured inside a function: measured at global scope they give false positives.
allocated_metric(st, x)         = (metric(st, x);         @allocated metric(st, x))
allocated_metric_inverse(st, x) = (metric_inverse(st, x); @allocated metric_inverse(st, x))
allocated_christoffel(st, x)    = (christoffel(st, x);    @allocated christoffel(st, x))
allocated_coordinate_radius(st, x) = (coordinate_radius(st, x); @allocated coordinate_radius(st, x))

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
    @test allocated_coordinate_radius(spacetime, x) == 0
    @test coordinate_radius(spacetime, x) isa T
end

@testset "KerrSchildCoordinates - coordinate_radius (Kerr radius r)" begin
    a = 0.9
    spacetime = Kerr.KerrSchildCoordinates(1.0, a)
    r = 3.0
    # on the axis (x = y = 0): r = |z|
    @test coordinate_radius(spacetime, SVector(0.0, 0.0, 0.0, r)) ≈ r rtol=1e-14
    # on the equator (z = 0): x² + y² = r² + a²
    @test coordinate_radius(spacetime, SVector(0.0, sqrt(r^2 + a^2), 0.0, 0.0)) ≈ r rtol=1e-14
    # general point: x + iy = (r + ia) e^{iφ} sinθ, z = r cosθ
    θ, φ = 1.0, 0.5
    x = (r*cos(φ) - a*sin(φ)) * sin(θ)
    y = (r*sin(φ) + a*cos(φ)) * sin(θ)
    z = r * cos(θ)
    @test coordinate_radius(spacetime, SVector(0.0, x, y, z)) ≈ r rtol=1e-14
end
