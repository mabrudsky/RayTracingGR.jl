# ============================================ #
# Kerr parameters (Boyer-Lindquist metric,     #
# spherical coordinates q = (t, r, θ, φ))      #
# ============================================ #
struct BoyerLindquistCoordinates{T<:Union{Float32,Float64}} <: AbstractKerrSpacetime
    M::T
    a::T
    topology::SphericalCoordinate
    function BoyerLindquistCoordinates{T}(M, a) where {T}
        M >= zero(T) || throw(ArgumentError("M must be non-negative"))
        abs(a) <= M  || throw(ArgumentError("|a| must be ≤ M"))
        return new{T}(M, a, SphericalCoordinate())
    end
end
BoyerLindquistCoordinates(M::T, a::T) where {T<:Union{Float32,Float64}} = BoyerLindquistCoordinates{T}(M, a)  # positional: BoyerLindquistCoordinates(1.0, 0.5)
BoyerLindquistCoordinates(; M, a) = BoyerLindquistCoordinates(M, a)                                                # keywords:   BoyerLindquistCoordinates(M=1.0, a=0.5)
BoyerLindquistCoordinates(M::Real, a::Real) = throw(ArgumentError("Spacetime parameters must be Float32 or Float64 (both the same type); got M::$(typeof(M)), a::$(typeof(a))."))

numtype(::BoyerLindquistCoordinates{T}) where {T} = T   # precision of the parameters (checked against u0 in Ensemble_Problem)

# =============================================================================== #
# Internal auxiliary (not exported): calculates Σ = r² + a² cos²θ,                #
# Δ = r² - 2Mr + a² and sin²θ, shared between metric() and metric_inverse().      #
# =============================================================================== #
@inline function _Σ_Δ(spacetime::BoyerLindquistCoordinates, point::SVector{4,T}) where T
    t, r, θ, φ = point

    a = spacetime.a
    M = spacetime.M

    A2   = a^2
    r2   = r^2
    sin2 = sin(θ)^2
    cos2 = one(T) - sin2

    Σ = r2 + A2 * cos2
    Δ = r2 - 2*M*r + A2

    return Σ, Δ, sin2
end

# ======================================================================= #
# 1) metric(spacetime, point) -> g_{μν}   (point = (t, r, θ, φ))          #
#    Generic in T: accepts Float32/Float64 or Dual (for christoffel).     #
#                                                                         #
#    g_tt = -(1 - 2Mr/Σ)          g_tφ = -2Mar sin²θ / Σ                  #
#    g_rr = Σ/Δ                   g_θθ = Σ                                #
#    g_φφ = (r² + a² + 2Ma²r sin²θ / Σ) sin²θ                             #
# ======================================================================= #
@inline function metric(spacetime::BoyerLindquistCoordinates, point::SVector{4,T}) where T
    Σ, Δ, sin2 = _Σ_Δ(spacetime, point)
    r = point[2]
    a = spacetime.a
    M = spacetime.M

    g11 = -(one(T) - 2*M*r / Σ)
    g12 = zero(T)
    g13 = zero(T)
    g14 = -2*M*a*r * sin2 / Σ

    g21 = g12
    g22 = Σ / Δ
    g23 = zero(T)
    g24 = zero(T)

    g31 = g13
    g32 = g23
    g33 = Σ
    g34 = zero(T)

    g41 = g14
    g42 = g24
    g43 = g34
    g44 = (r^2 + a^2 + 2*M*a^2*r * sin2 / Σ) * sin2

    return SMatrix{4, 4, T, 16}(g11, g21, g31, g41,
                                g12, g22, g32, g42,
                                g13, g23, g33, g43,
                                g14, g24, g34, g44)
end

# ======================================================================= #
# 2) metric_inverse(spacetime, point) -> g^{μν}                           #
#                                                                         #
#    g^tt = -((r² + a²)² - a²Δ sin²θ) / (ΣΔ)     g^tφ = -2Mar / (ΣΔ)      #
#    g^rr = Δ/Σ                                  g^θθ = 1/Σ               #
#    g^φφ = (Δ - a² sin²θ) / (ΣΔ sin²θ)                                   #
# ======================================================================= #
@inline function metric_inverse(spacetime::BoyerLindquistCoordinates, point::SVector{4,T}) where T
    Σ, Δ, sin2 = _Σ_Δ(spacetime, point)
    r = point[2]
    a = spacetime.a
    M = spacetime.M

    ΣΔ = Σ * Δ

    g11 = -((r^2 + a^2)^2 - a^2 * Δ * sin2) / ΣΔ
    g12 = zero(T)
    g13 = zero(T)
    g14 = -2*M*a*r / ΣΔ

    g21 = g12
    g22 = Δ / Σ
    g23 = zero(T)
    g24 = zero(T)

    g31 = g13
    g32 = g23
    g33 = one(T) / Σ
    g34 = zero(T)

    g41 = g14
    g42 = g24
    g43 = g34
    g44 = (Δ - a^2 * sin2) / (ΣΔ * sin2)

    return SMatrix{4, 4, T, 16}(g11, g21, g31, g41,
                                g12, g22, g32, g42,
                                g13, g23, g33, g43,
                                g14, g24, g34, g44)
end

# ======================================================================= #
# 3) coordinate_radius(spacetime, point) -> Boyer-Lindquist radius r      #
#    It is a coordinate: point = (t, r, θ, φ).                            #
# ======================================================================= #
@inline coordinate_radius(spacetime::BoyerLindquistCoordinates, point::SVector{4,T}) where T = point[2]
