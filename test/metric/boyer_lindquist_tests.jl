using RayTracingGR
import Test: @testset, @test, @test_throws
import StaticArrays: SVector, SMatrix
import LinearAlgebra: I, det

# allocated_metric, allocated_metric_inverse, allocated_christoffel and allocated_coordinate_radius are defined in kerr_schild_tests.jl

# Jacobian ∂x_KS^μ / ∂x_BL^α of the change Boyer-Lindquist (t, r, θ, φ) -> Kerr-Schild (t, x, y, z):
#   x + i y = (r + i a) e^{iφ_KS} sinθ,   z = r cosθ,
#   dt_KS = dt + (2Mr/Δ) dr,              dφ_KS = dφ + (a/Δ) dr.
# The point is taken with φ_KS = φ and t_KS = t (integration constants set to zero).
function bl_to_ks(M, a, q::SVector{4,T}) where T
    t, r, θ, φ = q
    Δ = r^2 - 2*M*r + a^2
    sθ, cθ, sφ, cφ = sin(θ), cos(θ), sin(φ), cos(φ)

    x = (r*cφ - a*sφ) * sθ
    y = (r*sφ + a*cφ) * sθ
    z = r * cθ

    # columns: ∂/∂t, ∂/∂r (including dφ_KS/dr and dt_KS/dr), ∂/∂θ, ∂/∂φ
    dφ_dr = a / Δ
    dt_dr = 2*M*r / Δ
    J = SMatrix{4,4,T}(
        1, 0, 0, 0,
        dt_dr, cφ*sθ + dφ_dr*(-r*sφ - a*cφ)*sθ, sφ*sθ + dφ_dr*(r*cφ - a*sφ)*sθ, cθ,
        0, (r*cφ - a*sφ)*cθ, (r*sφ + a*cφ)*cθ, -r*sθ,
        0, (-r*sφ - a*cφ)*sθ, (r*cφ - a*sφ)*sθ, 0)
    return SVector{4,T}(t, x, y, z), J
end

@testset "BoyerLindquistCoordinates - constructors" begin
    @test Kerr.BoyerLindquistCoordinates(1.0, 0.5)     isa Kerr.BoyerLindquistCoordinates{Float64}
    @test Kerr.BoyerLindquistCoordinates(M=1.0, a=0.5) isa Kerr.BoyerLindquistCoordinates{Float64}
    @test Kerr.BoyerLindquistCoordinates(1f0, 0.5f0)   isa Kerr.BoyerLindquistCoordinates{Float32}
    @test Kerr.BoyerLindquistCoordinates(1.0, 0.5) == Kerr.BoyerLindquistCoordinates(M=1.0, a=0.5)
    @test Kerr.BoyerLindquistCoordinates(1.0, 0.5).topology isa RayTracingGR.SphericalCoordinate

    @test_throws ArgumentError Kerr.BoyerLindquistCoordinates(1, 0.5)      # Int is not allowed
    @test_throws ArgumentError Kerr.BoyerLindquistCoordinates(1.0, 0.5f0)  # mixed precision
    @test_throws ArgumentError Kerr.BoyerLindquistCoordinates(-1.0, 0.0)   # M < 0
    @test_throws ArgumentError Kerr.BoyerLindquistCoordinates(1.0, 2.0)    # |a| > M
end

@testset "BoyerLindquistCoordinates - isbits and zero allocations ($T)" for T in (Float64, Float32)
    spacetime = Kerr.BoyerLindquistCoordinates(T(1), T(0.9))
    q = SVector{4,T}(0.3, 3.0, 1.0, 0.5)

    @test isbitstype(typeof(spacetime))
    @test isbitstype(typeof(metric(spacetime, q)))
    @test isbitstype(typeof(metric_inverse(spacetime, q)))
    @test isbitstype(typeof(christoffel(spacetime, q)))
    @test eltype(christoffel(spacetime, q)) === T

    @test allocated_metric(spacetime, q) == 0
    @test allocated_metric_inverse(spacetime, q) == 0
    @test allocated_christoffel(spacetime, q) == 0
    @test allocated_coordinate_radius(spacetime, q) == 0
    @test coordinate_radius(spacetime, q) === q[2]
end

@testset "BoyerLindquistCoordinates - the metric is correct" begin
    M, a = 1.0, 0.9
    spacetime = Kerr.BoyerLindquistCoordinates(M, a)
    # outside the horizon, between the horizons (r₋ < r < r₊) and far away
    points = (SVector(0.0, 3.0, 1.0, 0.5), SVector(0.0, 1.2, 0.7, 2.0), SVector(0.0, 40.0, 2.5, 4.0))

    for q in points
        t, r, θ, φ = q
        g    = metric(spacetime, q)
        ginv = metric_inverse(spacetime, q)

        # 1) metric_inverse is the inverse of metric
        @test g * ginv ≈ I atol=1e-12

        # 2) known determinant: det g = -Σ² sin²θ
        Σ = r^2 + a^2 * cos(θ)^2
        @test det(g) ≈ -Σ^2 * sin(θ)^2 rtol=1e-12

        # 3) same spacetime as Kerr-Schild: g_BL = Jᵀ g_KS J
        x_ks, J = bl_to_ks(M, a, q)
        g_ks = metric(Kerr.KerrSchildCoordinates(M, a), x_ks)
        @test J' * g_ks * J ≈ g rtol=1e-12 atol=1e-12

        # 4) coordinate_radius gives the same Kerr radius r in both coordinate systems
        @test coordinate_radius(Kerr.KerrSchildCoordinates(M, a), x_ks) ≈ coordinate_radius(spacetime, q) rtol=1e-13
    end

    # 5) a = 0: Schwarzschild, diag(-(1 - 2M/r), 1/(1 - 2M/r), r², r² sin²θ)
    schwarzschild = Kerr.BoyerLindquistCoordinates(M, 0.0)
    q = SVector(0.0, 5.0, 1.1, 0.3)
    r, θ = q[2], q[3]
    f = 1 - 2M/r
    @test metric(schwarzschild, q) ≈ SMatrix{4,4}(-f, 0, 0, 0,  0, 1/f, 0, 0,  0, 0, r^2, 0,  0, 0, 0, r^2*sin(θ)^2) atol=1e-14
end
