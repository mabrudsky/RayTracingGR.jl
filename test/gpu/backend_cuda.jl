# ==================================================================================== #
# NVIDIA GPU (CUDA.jl), for the GPU test scripts. Defines the functions that depend    #
# on the GPU: the test scripts only call these (see backend_metal.jl for Apple GPUs).  #
# ==================================================================================== #
using CUDA

const GPU_NAME = "NVIDIA (CUDA.jl $(pkgversion(CUDA)))"

gpu_backend()     = CUDA.CUDABackend()
gpu_functional()  = CUDA.functional()
gpu_device_name() = CUDA.name(CUDA.device())

function gpu_memory_text()
    free = isdefined(CUDA, :free_memory) ? CUDA.free_memory() : CUDA.available_memory()   # CUDA.jl 6 / 5
    return string(round(free / 2^30, digits = 2), " GiB free of ", round(CUDA.total_memory() / 2^30, digits = 2), " GiB")
end

# Does the GPU code compiled by f() request memory while it runs (integrator escaping to the GPU heap)?
function kernel_allocates(f)
    ir = sprint(io -> CUDA.@device_code_llvm io = io dump_module = true f())
    return occursin("gpu_malloc", ir) || occursin("gpu_gc_pool_alloc", ir)
end
