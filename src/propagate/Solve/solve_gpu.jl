# ======================================================================= #
# GPU: GPUTsit5 / GPUVern7 / GPUVern9 (GPUTsit5 by default),              #
#   EnsembleGPUKernel(backend)                                            #
# One ray per GPU thread. The callback is the DiscreteCallback +          #
# bisection of boundary_callback_gpu.jl, so merge_callbacks = true.       #
# batch_size: rays sent to the GPU at a time (as in Skylight-Pulsar).     #
# GPUTsit5 by default (measured 2026-11-19 in Float32: same precision as  #
# GPUVern7 and GPUVern9 is worse; GPUTsit5 is the one tested on the A30). #
# ======================================================================= #
default_algorithm(::KA.GPU) = GPUTsit5()

function solve_ensemble(ep, alg::GPUODEAlgorithm, backend::KA.GPU, n, callback; options...)
    return SciMLBase.solve(ep, alg, EnsembleGPUKernel(backend); trajectories = n, callback = callback,
                           batch_size = 50_000, merge_callbacks = true, options...)
end
