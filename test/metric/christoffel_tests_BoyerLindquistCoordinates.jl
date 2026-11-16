using RayTracingGR
import Test: @testset, @test
import StaticArrays: SVector

# covariant_derivative_metric is defined in christoffel_tests_KerrSchildCoordinates.jl (included before this file)

# In Boyer-Lindquist the components grow like r² (g_φφ ≈ r² sin²θ), so the tolerance is relative to the
# size of the metric at each point: |∇g| ≤ tol · max|g|.
scale(spacetime, u) = maximum(abs.(metric(spacetime, u)))

@testset "Christoffel symbols - BoyerLindquistCoordinates" begin
    # Test case: Verify that the covariant derivative of the metric is zero for BoyerLindquistCoordinates
    # Points (t, r, θ, φ): outside the horizon, between the horizons (r₋ < r < r₊) and far away
    spacetime = Kerr.BoyerLindquistCoordinates(M=1.0, a=0.5)
    u0 = SVector(0.1, 3.0,   1.0, 0.5)
    u1 = SVector(0.1, 1.0,   0.7, 2.0)
    u2 = SVector(0.1, 34.79, 2.0, 4.0)

    ∇g0 = covariant_derivative_metric(spacetime, u0)
    ∇g1 = covariant_derivative_metric(spacetime, u1)
    ∇g2 = covariant_derivative_metric(spacetime, u2)
    println("===================================================")
    @test maximum(abs.(∇g0)) ≈ 0 atol=1e-12 * scale(spacetime, u0)
    println("Maximum value of |∇g0| (BL): ", maximum(abs.(∇g0)))
    @test maximum(abs.(∇g1)) ≈ 0 atol=1e-12 * scale(spacetime, u1)
    println("Maximum value of |∇g1| (BL): ", maximum(abs.(∇g1)))
    @test maximum(abs.(∇g2)) ≈ 0 atol=1e-12 * scale(spacetime, u2)
    println("Maximum value of |∇g2| (BL): ", maximum(abs.(∇g2)))
    println("===================================================")
end

@testset "Christoffel symbols - BoyerLindquistCoordinates (Float32)" begin
    spacetime = Kerr.BoyerLindquistCoordinates(M=1f0, a=0.5f0)
    u0 = SVector{4,Float32}(0.1, 3.0,   1.0, 0.5)
    u1 = SVector{4,Float32}(0.1, 1.0,   0.7, 2.0)
    u2 = SVector{4,Float32}(0.1, 34.79, 2.0, 4.0)

    ∇g0 = covariant_derivative_metric(spacetime, u0)
    ∇g1 = covariant_derivative_metric(spacetime, u1)
    ∇g2 = covariant_derivative_metric(spacetime, u2)
    @test eltype(christoffel(spacetime, u0)) === Float32
    @test maximum(abs.(∇g0)) ≈ 0 atol=1e-5 * scale(spacetime, u0)
    println("Maximum value of |∇g0| (BL, Float32): ", maximum(abs.(∇g0)))
    @test maximum(abs.(∇g1)) ≈ 0 atol=1e-5 * scale(spacetime, u1)
    println("Maximum value of |∇g1| (BL, Float32): ", maximum(abs.(∇g1)))
    @test maximum(abs.(∇g2)) ≈ 0 atol=1e-5 * scale(spacetime, u2)
    println("Maximum value of |∇g2| (BL, Float32): ", maximum(abs.(∇g2)))
    println("===================================================")
end
