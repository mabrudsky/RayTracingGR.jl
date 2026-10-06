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