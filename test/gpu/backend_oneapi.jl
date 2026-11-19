# ==================================================================================== #
# Intel GPU (oneAPI.jl, Linux), for the GPU test scripts. Same functions as            #
# backend_cuda.jl. Some Intel GPUs have no Float64: then everything on the GPU is      #
# Float32 (KernelAbstractions.supports_float64 tells).                                 #
# NOT TESTED YET on an Intel GPU (written 2026-11-19): if a function fails, the        #
# scripts print "could not be checked" (or "(unknown)") for that part and go on.       #
# ==================================================================================== #
using oneAPI

const GPU_NAME = "Intel (oneAPI.jl $(pkgversion(oneAPI)))"

gpu_backend()     = oneAPI.oneAPIBackend()
gpu_functional()  = oneAPI.functional()
gpu_device_name() = string(oneAPI.properties(oneAPI.device()).name)

function gpu_memory_text()
    total = sum(p.totalSize for p in oneAPI.memory_properties(oneAPI.device()))
    return string(round(total / 2^30, digits = 2), " GiB total")
end

# Does the GPU code compiled by f() request memory while it runs (integrator escaping to the GPU heap)?
function kernel_allocates(f)
    ir = sprint(io -> oneAPI.@device_code_llvm io = io dump_module = true f())
    return occursin("gpu_malloc", ir) || occursin("gpu_gc_pool_alloc", ir)
end
