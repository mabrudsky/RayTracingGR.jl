# ==================================================================================== #
# AMD GPU (AMDGPU.jl, ROCm on Linux), for the GPU test scripts. Same functions as      #
# backend_cuda.jl.                                                                     #
# NOT TESTED YET on an AMD GPU (written 2026-11-19): if a function fails, the scripts  #
# print "could not be checked" (or "(unknown)") for that part and go on.               #
# ==================================================================================== #
using AMDGPU

const GPU_NAME = "AMD (AMDGPU.jl $(pkgversion(AMDGPU)))"

gpu_backend()     = AMDGPU.ROCBackend()
gpu_functional()  = AMDGPU.functional()
gpu_device_name() = string(AMDGPU.device())

function gpu_memory_text()
    free, total = AMDGPU.Runtime.Mem.info()
    return string(round(free / 2^30, digits = 2), " GiB free of ", round(total / 2^30, digits = 2), " GiB")
end

# Does the GPU code compiled by f() request memory while it runs (integrator escaping to the GPU heap)?
function kernel_allocates(f)
    ir = sprint(io -> AMDGPU.@device_code_llvm io = io dump_module = true f())
    return occursin("gpu_malloc", ir) || occursin("gpu_gc_pool_alloc", ir)
end
