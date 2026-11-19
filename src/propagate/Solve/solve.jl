# ================================================================================ #
# propagate(configuration, initial_data, region; alg, options...)                  #
# integrates every ray (one column of initial_data) until it leaves the region,    #
# and returns the final states with the same layout: N × n, one column per ray.    #
# Everything is chosen from configuration.backend (one file per backend):          #
#                    default alg      ensemble                                     #
#   CPU()         -> Vern9()          EnsembleThreads()          (solve_cpu.jl)    #
#   GPU           -> GPUTsit5()       EnsembleGPUKernel(backend) (solve_gpu.jl)    #
# The algorithm and the options of solve can be changed with keywords, e.g.        #
#   propagate(configuration, initial_data, region; alg = Tsit5(), reltol = 1e-10) #
# All the arguments can also be given by name, like in Configuration:              #
#   propagate(configuration = ..., initial_data = ..., region = ...)               #
# trajectories and callback are set here; batch_size and merge_callbacks in        #
# solve_gpu.jl. They can not be changed.                                           #
# ================================================================================ #

#= Declare "default_algorithm" and "solve_ensemble" here first, with no
   implementation yet, so that every backend adds to these SAME functions.
=#
function default_algorithm end   # default_algorithm(backend) -> integration algorithm used by default
function solve_ensemble end      # solve_ensemble(ep, alg, backend, n, callback; options...) -> solution of the n rays

#------------------------------- backends ---------------------------------
include("solve_cpu.jl")
include("solve_gpu.jl")

# Any other combination: the algorithm does not run on that backend
solve_ensemble(ep, alg, backend, n, callback; options...) =
    throw(ArgumentError("$(nameof(typeof(alg))) can not be used with the backend $(nameof(typeof(backend))): " *
                        "on the CPU use Tsit5, Vern7 or Vern9; on a GPU use GPUTsit5, GPUVern7 or GPUVern9"))

#------------------------------- tolerances -------------------------------
# Smallest tolerances the precision allows (the boundary is interpolated by the callback).
# Measured (2026-11-19, pulsar in Kerr-Schild): with 5 eps(T) the error still decreases and
# no ray fails; below it the error does not improve.
default_tolerance(::Type{T}) where {T <: Union{Float32,Float64}} = 5 * eps(T)

# Options of solve that propagate sets itself
const INTERNAL_OPTIONS = (:trajectories, :callback, :batch_size, :merge_callbacks)

function propagate(configuration::Configuration{N,T}, initial_data::AbstractMatrix, region::AbstractRegion;
                   alg = default_algorithm(configuration.backend), kwargs...) where {N, T}
    for key in keys(kwargs)
        key in INTERNAL_OPTIONS && throw(ArgumentError("$key is set by propagate and can not be changed"))
    end
    # Default options of solve; the keywords of the user replace them
    options = (; abstol = default_tolerance(T), reltol = default_tolerance(T), adaptive = true,
                 save_everystep = false, save_start = false, kwargs...)

    ep  = Ensemble_Problem(configuration, initial_data)
    cb  = boundary_callback(region, configuration.backend)
    sol = solve_ensemble(ep, alg, configuration.backend, size(initial_data, 2), cb; options...)
    return stack(sol.u)   # N × n: final state of each ray, one column per ray
end

# Keywords, like Configuration: propagate(configuration = ..., initial_data = ..., region = ..., alg = ..., reltol = ...)
propagate(; configuration::Configuration, initial_data::AbstractMatrix, region::AbstractRegion, kwargs...) =
    propagate(configuration, initial_data, region; kwargs...)
