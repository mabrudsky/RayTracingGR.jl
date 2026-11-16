# ================================================================================ #
# Integration regions: each scenario (pulsar, accretion disk, ...) is a region     #
# where the ray is integrated, and defines                                         #
#   region_condition(region, spacetime, position) -> > 0 while the ray is inside   #
#   the region, and changes sign at its boundary.                                  #
# boundary_callback(region) turns it into a ContinuousCallback that finds the      #
# crossing and stops the ray there (terminate!). The same callback works on the    #
# CPU (Tsit5, EnsembleThreads) and on the GPU (GPUTsit5, EnsembleGPUKernel, with   #
# merge_callbacks = true); only the final state is saved.                          #
# ================================================================================ #

# ======================================================================= #
# Pulsar: the ray is integrated while R_in < r < R_out, where r is the    #
# coordinate_radius of the spacetime (e.g. the Kerr radius).              #
#   R_in  : radius of the neutron star (the ray hits the star)            #
#   R_out : outer radius (the ray escapes towards the observer)           #
# ======================================================================= #
struct PulsarRegion{T<:Union{Float32,Float64}} <: AbstractRegion
    R_in::T
    R_out::T
    function PulsarRegion{T}(R_in, R_out) where {T}
        R_in > zero(T) || throw(ArgumentError("R_in must be positive"))
        R_out > R_in   || throw(ArgumentError("R_out must be greater than R_in"))
        return new{T}(R_in, R_out)
    end
end
PulsarRegion(R_in::T, R_out::T) where {T<:Union{Float32,Float64}} = PulsarRegion{T}(R_in, R_out)  # positional: PulsarRegion(5.0, 50.0)
PulsarRegion(; R_in, R_out) = PulsarRegion(R_in, R_out)                                           # keywords:   PulsarRegion(R_in=5.0, R_out=50.0)
PulsarRegion(R_in::Real, R_out::Real) = throw(ArgumentError("Region parameters must be Float32 or Float64 (both the same type); got R_in::$(typeof(R_in)), R_out::$(typeof(R_out))."))

# (r - R_in)(R_out - r): positive inside, zero at the boundary, negative outside
@inline function region_condition(region::PulsarRegion, spacetime::AbstractSpacetime, position::SVector{4,T}) where T
    r = coordinate_radius(spacetime, position)
    return (r - region.R_in) * (region.R_out - r)
end

# ======================================================================= #
# ContinuousCallback + terminate! at the boundary of the region           #
# (CPU and GPU). The spacetime is read from p = IntegrationParameters.   #
# ======================================================================= #
function boundary_callback(region::AbstractRegion)
    condition(u, t, integrator) = region_condition(region, integrator.p.spacetime, SVector(u[1], u[2], u[3], u[4]))
    return SciMLBase.ContinuousCallback(condition, SciMLBase.terminate!; save_positions = (false, false))
end
