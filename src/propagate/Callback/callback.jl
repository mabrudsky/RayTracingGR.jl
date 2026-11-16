# ================================================================================ #
# Integration regions: each scenario (pulsar, accretion disk, ...) is a region     #
# where the ray is integrated, and defines                                         #
#   region_condition(region, spacetime, position) -> > 0 while the ray is inside   #
#   the region, and changes sign at its boundary.                                  #
# boundary_callback(region, backend) turns it into a callback that finds the       #
# crossing and stops the ray there (terminate!). The type of callback is chosen    #
# from the backend (only the final state is saved):                                #
#   CPU() -> ContinuousCallback (Tsit5, EnsembleThreads)                           #
#   GPU   -> DiscreteCallback + bisection on the interpolation of the integrator   #
#            (GPUTsit5, EnsembleGPUKernel, merge_callbacks = true)                 #
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
# boundary_callback: the callback is chosen from the backend              #
#   boundary_callback(region, configuration)  (uses configuration.backend) #
#   boundary_callback(region, backend)                                    #
# The spacetime is read from the parameters p = IntegrationParameters.   #
# ======================================================================= #
boundary_callback(region::AbstractRegion, configuration::Configuration) = boundary_callback(region, configuration.backend)

# CPU: ContinuousCallback + terminate!
function boundary_callback(region::AbstractRegion, ::KA.CPU)
    condition(u, t, integrator) = region_condition(region, integrator.p.spacetime, SVector(u[1], u[2], u[3], u[4]))
    return SciMLBase.ContinuousCallback(condition, SciMLBase.terminate!; save_positions = (false, false))
end

# GPU: DiscreteCallback + bisection on the interpolation of the integrator + terminate!
#   ContinuousCallback makes the integrator escape to the GPU heap in EnsembleGPUKernel
#   ("Out of dynamic GPU memory", DiffEqGPU issue #219). Every function that receives the
#   integrator is @inline, so the integrator stays in registers.
#   At the end of each step: if the ray left the region, the crossing time is found by
#   bisection between tprev (inside) and t (outside) using the interpolation integrator(t),
#   the state is moved there and the integration stops.
function boundary_callback(region::AbstractRegion, ::KA.GPU)
    @inline inside(u, integrator) = region_condition(region, integrator.p.spacetime, SVector(u[1], u[2], u[3], u[4]))
    @inline condition(u, t, integrator) = inside(u, integrator) <= 0
    @inline function affect!(integrator)
        t_in, t_out = integrator.tprev, integrator.t
        for _ in 1:50
            t_mid = (t_in + t_out) / 2
            inside(integrator(t_mid), integrator) > 0 ? (t_in = t_mid) : (t_out = t_mid)
        end
        SciMLBase.change_t_via_interpolation!(integrator, t_out)
        SciMLBase.terminate!(integrator)
    end
    return SciMLBase.DiscreteCallback(condition, affect!; save_positions = (false, false))
end
