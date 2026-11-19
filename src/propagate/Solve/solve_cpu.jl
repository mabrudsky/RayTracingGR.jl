# ======================================================================= #
# CPU: Tsit5 / Vern7 / Vern9 (Vern9 by default), EnsembleThreads()       #
# One ray per thread. The callback is the ContinuousCallback of           #
# boundary_callback_cpu.jl.                                               #
# ======================================================================= #
default_algorithm(::KA.CPU) = Vern9()

function solve_ensemble(ep, alg::SciMLBase.AbstractODEAlgorithm, ::KA.CPU, n, callback; options...)
    return SciMLBase.solve(ep, alg, SciMLBase.EnsembleThreads(); trajectories = n, callback = callback, options...)
end

# The algorithms of DiffEqGPU are also AbstractODEAlgorithm, but only run with EnsembleGPUKernel
solve_ensemble(ep, alg::GPUODEAlgorithm, backend::KA.CPU, n, callback; options...) =
    throw(ArgumentError("$(nameof(typeof(alg))) can not be used with the backend CPU: use Tsit5, Vern7 or Vern9"))
