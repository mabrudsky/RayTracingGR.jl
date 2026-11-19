# ======================================================================= #
# Accretion disk around a black hole: thin disk on the equatorial plane   #
# z = 0 between R_in and R_out. The ray is integrated until:              #
#   - it crosses the equatorial plane inside the disk (R_in ≤ r ≤ R_out): #
#     it hits the disk;                                                   #
#   - r = R_min: it falls into the black hole (e.g. R_min = r_+ + 0.05);  #
#   - r = R_max: it escapes.                                              #
# Crossing the equatorial plane outside the disk does NOT stop the ray.   #
#   r : coordinate_radius (Kerr radius)                                   #
#   z : coordinate_z, height above the equatorial plane (z = r cos θ)     #
# R_min is given by the user: the event horizon is not a local quantity.  #
# ======================================================================= #
struct AccretionDiskRegion{T<:Union{Float32,Float64}} <: AbstractRegion
    R_min::T
    R_max::T
    R_in::T
    R_out::T
    function AccretionDiskRegion{T}(R_min, R_max, R_in, R_out) where {T}
        R_min >= zero(T) || throw(ArgumentError("R_min must be non-negative"))
        R_in >= R_min    || throw(ArgumentError("R_in must be greater than or equal to R_min"))
        R_out >= R_in    || throw(ArgumentError("R_out must be greater than or equal to R_in"))
        R_max > R_out    || throw(ArgumentError("R_max must be greater than R_out"))
        return new{T}(R_min, R_max, R_in, R_out)
    end
end
AccretionDiskRegion(R_min::T, R_max::T, R_in::T, R_out::T) where {T<:Union{Float32,Float64}} =
    AccretionDiskRegion{T}(R_min, R_max, R_in, R_out)                    # positional: AccretionDiskRegion(2.0, 6000.0, 2.0, 15.0)
AccretionDiskRegion(; R_min, R_max, R_in, R_out) = AccretionDiskRegion(R_min, R_max, R_in, R_out)   # keywords
AccretionDiskRegion(R_min::Real, R_max::Real, R_in::Real, R_out::Real) =
    throw(ArgumentError("Region parameters must be Float32 or Float64 (all the same type); got " *
                        "$(typeof(R_min)), $(typeof(R_max)), $(typeof(R_in)), $(typeof(R_out))."))

# (r - R_min)(R_max - r) z: changes sign at r = R_min, at r = R_max and on the equatorial plane
@inline function region_condition(region::AccretionDiskRegion, spacetime::AbstractSpacetime, position::SVector{4,T}) where T
    r = coordinate_radius(spacetime, position)
    z = coordinate_z(spacetime, position)
    return (r - region.R_min) * (region.R_max - r) * z
end

# At a sign change of the condition: on the equatorial plane (z closer to 0 than r to R_min or R_max)
# the ray stops only inside the disk; at r = R_min or r = R_max it always stops.
@inline function stops_at_boundary(region::AccretionDiskRegion, spacetime::AbstractSpacetime, position::SVector{4,T}) where T
    r = coordinate_radius(spacetime, position)
    z = coordinate_z(spacetime, position)
    on_plane = abs(z) < min(r - region.R_min, region.R_max - r)
    return !on_plane || (region.R_in <= r <= region.R_out)
end
