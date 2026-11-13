# Benchmarks (not part of the tests). Run from the project folder:
#   julia --project=benchmark benchmark/benchmarks.jl
using RayTracingGR
using BenchmarkTools
import ForwardDiff: Dual, Partials, partials
import StaticArrays: SVector, @SArray

# Same function as in test/metric/christoffel_tests_KerrSchildCoordinates.jl
function covariant_derivative_metric(k, u::SVector{4,T}) where T
    Γ = christoffel(k, u)
    g = metric(k, u)

    xd = SVector{4, Dual{Nothing,T,4}}(ntuple(i -> Dual{Nothing,T,4}(u[i], Partials(ntuple(j -> T(i == j), 4))), 4))
    gd = metric(k, xd)
    ∂g = partials.(gd)

    # ES: ∇_σ g_{μν} = ∂_σ g_{μν} - Γ^ρ_{σμ} g_{ρν} - Γ^ρ_{σν} g_{μρ}, devuelto como SArray inmutable.
    # EN: ∇_σ g_{μν} = ∂_σ g_{μν} - Γ^ρ_{σμ} g_{ρν} - Γ^ρ_{σν} g_{μρ}, returned as an immutable SArray.
    return @SArray T[∂g[μ, ν][σ] - sum(Γ[σ, μ, :] .* g[:, ν]) - sum(Γ[σ, ν, :] .* g[μ, :]) for σ in 1:4, μ in 1:4, ν in 1:4]
end

spacetime = Kerr.KerrSchildCoordinates(M=1.0, a=0.5)
u0 = @SArray [0.1, 3.0, 0.05, 1.0]
u1 = 0.75 .* u0
u2 = 34.79 .* u0

print("∇g0: "); @btime covariant_derivative_metric($spacetime, $u0)
print("∇g1: "); @btime covariant_derivative_metric($spacetime, $u1)
print("∇g2: "); @btime covariant_derivative_metric($spacetime, $u2)

println("=============== Benchmarks (@btime) ===============")
print(rpad("metric:", 32));                      @btime metric($spacetime, $u0)
print(rpad("metric_inverse:", 32));              @btime metric_inverse($spacetime, $u0)
print(rpad("christoffel:", 32));                 @btime christoffel($spacetime, $u0)
print(rpad("covariant_derivative_metric:", 32)); @btime covariant_derivative_metric($spacetime, $u0)
println("===================================================")
