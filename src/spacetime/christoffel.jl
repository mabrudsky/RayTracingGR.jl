abstract type AbstractChristoffel end
struct ChristoffelTag <: AbstractChristoffel end

""" Declare "christoffel" here first, with no implementation yet,
    so that every spacetime module (Kerr, Schwarzschild, etc.) adds
    to this SAME function instead of each one creating its own.
"""
function christoffel end
#============================================================================================================#
@inline function _Γcomp(ginv::SMatrix{4,4,T}, ∂g, λ::Int, μ::Int, ν::Int) where T
    s = zero(T)
    @inbounds for α in 1:4
        # ∂g[α,ν][μ] is ∂(g_αν)/∂x^μ
        s += ginv[λ, α] * (∂g[α, ν][μ] + ∂g[α, μ][ν] - ∂g[μ, ν][α])
    end
    return T(0.5) * s
end


@inline function _christoffel_ForwardDiff(k::AbstractSpacetime, position::SVector{4,T}) where T
    xd   = SVector{4, Dual{ChristoffelTag,T,4}}(ntuple(i -> Dual{ChristoffelTag,T,4}(position[i], Partials(ntuple(j -> T(i == j), 4))), 4))
    gd   = metric(k, xd)    # SMatrix{4,4,Dual}
    ∂g   = partials.(gd)    # ∂g[μ,ν][σ] = ∂g_{μν}/∂x^σ
    ginv = metric_inverse(k, position)
    return @SArray [_Γcomp(ginv, ∂g, λ, μ, ν) for μ in 1:4, ν in 1:4, λ in 1:4]
end


#=
@inline function _christoffel_ForwardDiff(k::AbstractSpacetime, position::SVector{4,T}) where T
    xd   = SVector{4, Dual{ChristoffelTag,T,4}}(ntuple(i -> Dual{ChristoffelTag,T,4}(position[i], Partials(ntuple(j -> T(i == j), 4))), 4))
    gd   = metric(k, xd)
    ginv = metric_inverse(k, position)

    # Paso 1: dg[α,β,σ] = ∂g_{αβ}/∂x^σ  (números normales)
    dg = @SArray T[partials(gd[α, β], σ) for α in 1:4, β in 1:4, σ in 1:4]

    # Paso 2: Γlow[α,μ,ν] = ½(∂_μ g_{αν} + ∂_ν g_{αμ} − ∂_α g_{μν})   ← se calcula UNA vez
    Γlow = @SArray T[T(0.5) * (dg[α, ν, μ] + dg[α, μ, ν] - dg[μ, ν, α])
                     for α in 1:4, μ in 1:4, ν in 1:4]

    # Paso 3: Γ^λ_{μν} = Σ_α g^{λα} Γlow[α,μ,ν]
    return @SArray T[sum(ginv[λ, :] .* Γlow[:, μ, ν]) for μ in 1:4, ν in 1:4, λ in 1:4]
end
=#