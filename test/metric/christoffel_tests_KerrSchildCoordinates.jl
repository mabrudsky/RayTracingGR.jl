using RayTracingGR
using BenchmarkTools
import Test: @testset, @test
import ForwardDiff: Dual, Partials, partials
import StaticArrays: SVector, @SVector, @SArray, @MArray
#=
"""
   ES: `&&` ejecuta el return si la condición es VERDADERA. `||` ejecuta el return si la condición es FALSA.
   EN: `&&` runs the return if the condition is TRUE. `||` runs the return if the condition is FALSE.
"""
function fmt_time(t)
    t < 1e-6 && return string(round(t * 1e9, digits=1), " ns")
    t < 1e-3 && return string(round(t * 1e6, digits=2), " μs")
    t < 1    && return string(round(t * 1e3, digits=2), " ms")
    return string(round(t, digits=3), " s")
end
=#
    function covariant_derivative_metric(k, u::SVector{4,T}) where T
        Γ = christoffel(k, u)
        g = metric(k, u)

        xd = SVector{4, Dual{Nothing,T,4}}(ntuple(i -> Dual{Nothing,T,4}(u[i], Partials(ntuple(j -> T(i == j), 4))), 4))
        gd = metric(k, xd)
        ∂g = partials.(gd)

        # ES: ∇_σ g_{μν} = ∂_σ g_{μν} - Γ^ρ_{σμ} g_{ρν} - Γ^ρ_{σν} g_{μρ}, devuelto como SArray inmutable.
        # EN: ∇_σ g_{μν} = ∂_σ g_{μν} - Γ^ρ_{σμ} g_{ρν} - Γ^ρ_{σν} g_{μρ}, returned as an immutable SArray.
        return @SArray T[∂g[μ, ν][σ] - sum(Γ[σ, μ, :] .* g[:, ν]) - sum(Γ[σ, ν, :] .* g[μ, :])
                         for σ in 1:4, μ in 1:4, ν in 1:4]
        #return ∇g
    end




@testset "Christoffel symbols - KerrSchildCoordinates" begin
    # Test case: Verify that the covariant derivative of the metric is zero for KerrSchildCoordinates
    spacetime = kerr.KerrSchildCoordinates(M=1.0, a=0.5)
    u0 = @SArray [0.1, 3.0, 0.05, 1.0]
    u1 = 0.75 .* u0
    u2 = 34.79 .* u0

    print("∇g0: "); ∇g0 = @btime covariant_derivative_metric($spacetime, $u0)
    print("∇g1: "); ∇g1 = @btime covariant_derivative_metric($spacetime, $u1)
    print("∇g2: "); ∇g2 = @btime covariant_derivative_metric($spacetime, $u2)
    println("===================================================")
    # Verify that the maximum value of |∇g| is zero within floating-point tolerance
    @test maximum(abs.(∇g0)) ≈ 0 atol=1e-12  # Equivalent to @test isapprox(maximum(abs.(∇g0)), 0; atol=1e-12)
    println("Maximum value of |∇g0|: ", maximum(abs.(∇g0)))
    @test maximum(abs.(∇g1)) ≈ 0 atol=1e-12  # Equivalent to @test isapprox(maximum(abs.(∇g1)), 0; atol=1e-12)
    println("Maximum value of |∇g1|: ", maximum(abs.(∇g1)))
    @test maximum(abs.(∇g2)) ≈ 0 atol=1e-12  # Equivalent to @test isapprox(maximum(abs.(∇g2)), 0; atol=1e-12)
    println("Maximum value of |∇g2|: ", maximum(abs.(∇g2)))

    println("=============== Benchmarks (@btime) ===============")
    print(rpad("metric:", 32));                      @btime metric($spacetime, $u0)
    print(rpad("metric_inverse:", 32));              @btime metric_inverse($spacetime, $u0)
    print(rpad("christoffel:", 32));                 @btime christoffel($spacetime, $u0)
    print(rpad("covariant_derivative_metric:", 32)); @btime covariant_derivative_metric($spacetime, $u0)
    println("===================================================")
end

    