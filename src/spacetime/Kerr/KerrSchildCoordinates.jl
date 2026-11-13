# ======================================== #
# Kerr parameters (Kerr-Schild metric,     #  
# Cartesian coordinates q = (t, x, y, z))  #
# ======================================== #
struct KerrSchildCoordinates{T<:Union{Float32,Float64}} <: AbstractKerrSpacetime
    M::T
    a::T
    topology::CartesianCoordinate
    function KerrSchildCoordinates{T}(M, a) where {T}
        M >= zero(T) || throw(ArgumentError("M must be non-negative"))
        abs(a) <= M  || throw(ArgumentError("|a| must be ≤ M"))
        return new{T}(M, a, CartesianCoordinate())
    end
end
KerrSchildCoordinates(M::T, a::T) where {T<:Union{Float32,Float64}} = KerrSchildCoordinates{T}(M, a)  # positional: KerrSchildCoordinates(1.0, 0.5)
KerrSchildCoordinates(; M, a) = KerrSchildCoordinates(M, a)                                            # keywords:   KerrSchildCoordinates(M=1.0, a=0.5)
KerrSchildCoordinates(M::Real, a::Real) = throw(ArgumentError("Spacetime parameters must be Float32 or Float64 (both the same type); got M::$(typeof(M)), a::$(typeof(a))."))

numtype(::KerrSchildCoordinates{T}) where {T} = T   # precision of the parameters (checked against u0 in Ensemble_Problem)

# =============================================================================== # 
# Internal auxiliary (not exported): calculates the scalar H and the null vector  #
# l, shared between metric() and metric_inverse().                                #
# =============================================================================== #
@inline function _H_l(spacetime::KerrSchildCoordinates, point::SVector{4,T}) where T
    t, x, y, z = point
    
    a = spacetime.a
    M = spacetime.M

    A2   = a^2
    rho2 = x^2 + y^2 + z^2
    r2   = T(0.5) * (rho2 - A2) + sqrt(T(0.25) * (rho2 - A2)^2 + A2 * z^2)
    r    = sqrt(r2)
    _2H  = 2*M*r / (r2 + A2 * z^2 / r2)

    l1 = one(T)
    l2 = (r*x + a*y) / (r2 + A2)
    l3 = (r*y - a*x) / (r2 + A2)
    l4 = z / r

    return _2H, SVector{4,T}(l1, l2, l3, l4)
end
   
# ==================================================================== #
# 1) metric(k, point) -> g_{μν}   (point = (t, x, y, z))               #  
#    Generic in T: accepts Float32/Float64 or Dual (for christoffel).  #
# ==================================================================== #
@inline function metric(spacetime::KerrSchildCoordinates, point::SVector{4,T}) where T
    _2H, l = _H_l(spacetime, point)
    
    g11 = - one(T) + _2H * l[1]*l[1]
    g12 = _2H * l[1]*l[2]
    g13 = _2H * l[1]*l[3]
    g14 = _2H * l[1]*l[4]
    
    g21 = g12
    g22 = one(T) + _2H * l[2]*l[2]
    g23 = _2H * l[2]*l[3]
    g24 = _2H * l[2]*l[4]
    
    g31 = g13
    g32 = g23
    g33 = one(T) + _2H * l[3]*l[3]
    g34 = _2H * l[3]*l[4]
    
    g41 = g14
    g42 = g24
    g43 = g34
    g44 = one(T) + _2H * l[4]*l[4]    

    return SMatrix{4, 4, T, 16}(g11, g21, g31, g41, 
                                g12, g22, g32, g42, 
                                g13, g23, g33, g43, 
                                g14, g24, g34, g44)
end

# =============================================================== #
# 2) metric_inverse(k, point) -> g^{μν} = η^{μν} - 2H l^μ l^ν     #
#  with l^μ = η^{μν} l_ν = (-1, l^i) (η^{μν} = diag(-1, 1, 1, 1)) #
# =============================================================== #
@inline function metric_inverse(spacetime::KerrSchildCoordinates, point::SVector{4,T}) where T
    _2H, l = _H_l(spacetime, point)

    g11 = -one(T) - _2H * l[1] * l[1]
    g12 =  _2H * l[1] * l[2]
    g13 =  _2H * l[1] * l[3]
    g14 =  _2H * l[1] * l[4]
    
    g21 = g12
    g22 = one(T) - _2H * l[2] * l[2]
    g23 = - _2H * l[2] * l[3]
    g24 = - _2H * l[2] * l[4]
    
    g31 = g13
    g32 = g23
    g33 = one(T) - _2H * l[3] * l[3]
    g34 = - _2H * l[3] * l[4]
    
    g41 = g14
    g42 = g24
    g43 = g34
    g44 = one(T) - _2H * l[4] * l[4]


    return SMatrix{4, 4, T, 16}(g11, g21, g31, g41, 
                                g12, g22, g32, g42, 
                                g13, g23, g33, g43, 
                                g14, g24, g34, g44)
end
