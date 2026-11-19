# ==================================================================================== #
# Apple GPU (Metal.jl, macOS with Apple silicon: M1, M2, ...), for the GPU test        #
# scripts. Same functions as backend_cuda.jl.                                          #
# Metal has no Float64: everything on the GPU is Float32 (the CPU reference stays in   #
# Float64).                                                                            #
# NOT TESTED YET on a Mac (written 2026-11-19): if a function fails, the scripts print #
# "could not be checked" for that part and go on.                                      #
# ==================================================================================== #
using Metal

const GPU_NAME = "Apple (Metal.jl $(pkgversion(Metal)))"

gpu_backend()     = Metal.MetalBackend()
gpu_functional()  = Metal.functional()
gpu_device_name() = String(Metal.device().name)

function gpu_memory_text()
    dev = Metal.device()
    return string(round(dev.recommendedMaxWorkingSetSize / 2^30, digits = 2), " GiB recommended working set")
end

# Does the GPU code compiled by f() request memory while it runs (integrator escaping to the GPU heap)?
function kernel_allocates(f)
    ir = sprint(io -> Metal.@device_code_llvm io = io dump_module = true f())
    return occursin("gpu_malloc", ir) || occursin("gpu_gc_pool_alloc", ir)
end
