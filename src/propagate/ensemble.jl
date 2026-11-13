struct GeodesicEquationsVacuum <: AbstractEquationSet end
struct ParallelTransportEquationsVacuum <: AbstractEquationSet end

# Number of components of the state vector u required by each equation set
state_length(::GeodesicEquationsVacuum)          = 8
state_length(::ParallelTransportEquationsVacuum) = 16
# Right-hand side (ODE function) of each equation set
rhs(::GeodesicEquationsVacuum)          = geodesic_equations_Vacuum
rhs(::ParallelTransportEquationsVacuum) = parallel_transport_equations_Vacuum
#-------------------------------------------------------------------------
function Ensemble_Problem(configuration::Configuration{N,T,S,E,B}, initial_data::AbstractMatrix) where {N, T <: Union{Float32,Float64}, S <: AbstractSpacetime, E <: AbstractEquationSet, B <: KA.Backend}
    N == state_length(configuration.equations) || throw(DimensionMismatch("$(nameof(E)) needs u0 of length $(state_length(configuration.equations)), got $N"))
    numtype(configuration.spacetime) === T || throw(ArgumentError("spacetime uses $(numtype(configuration.spacetime)) but u0 uses $T; use the same precision (e.g. KerrSchildCoordinates(M = 1f0, a = 0.5f0) for Float32)"))
    size(initial_data, 1) == N || throw(DimensionMismatch("Each column of initial_data must be an initial condition. Expected size $N, got $(size(initial_data, 1))"))
    eltype(initial_data) === T || throw(ArgumentError("initial_data must have element type $T"))

    (; u0, spacetime, tspan, equations) = configuration
    p           = IntegrationParameters(spacetime)
    prob        = SciMLBase.ODEProblem{false}(rhs(equations), u0, tspan, p)
    prob_func   = (prob, ctx) -> SciMLBase.remake(prob, u0 = SVector{N, T}(@view initial_data[:,ctx.sim_id]))
    output_func = (sol, ctx) -> (sol.u[end], false)
    return SciMLBase.EnsembleProblem(prob, prob_func=prob_func, output_func=output_func, safetycopy=false)
end
