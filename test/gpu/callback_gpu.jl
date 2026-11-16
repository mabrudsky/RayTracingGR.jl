# ==================================================================================== #
# Test of boundary_callback on a real GPU.                                             #
# Not part of Pkg.test() (the CI has no GPU). Run from the project folder:             #
#   julia --project=test/gpu -e "using Pkg; Pkg.instantiate()"     # first time only   #
#   julia -t auto --project=test/gpu test/gpu/callback_gpu.jl                          #
# Options (environment variables):                                                     #
#   NRAYS          = "10000,100000,1000000"  number of rays of each run                #
#   NCPU           = "10000"                 rays compared against the CPU (Tsit5)     #
#   TRY_CONTINUOUS = "false"                 also try ContinuousCallback on the GPU    #
#                                            (known to fail: DiffEqGPU issue #219)     #
# With a GPU backend, boundary_callback(region, backend) gives a DiscreteCallback +    #
# bisection on the interpolation (all @inline, so the integrator does not escape to    #
# the GPU heap). Without a working NVIDIA GPU the same GPU kernel runs on the CPU.     #
# ==================================================================================== #
using RayTracingGR, StaticArrays, KernelAbstractions, SciMLBase, DiffEqGPU, OrdinaryDiffEqTsit5, CUDA
using Random, Printf
const RT = RayTracingGR

const USE_GPU        = CUDA.functional()
const BACKEND        = USE_GPU ? CUDA.CUDABackend() : KernelAbstractions.CPU()
const NRAYS          = parse.(Int, split(get(ENV, "NRAYS", "10000,100000,1000000"), ","))
const NCPU           = parse(Int, get(ENV, "NCPU", "10000"))
const TRY_CONTINUOUS = get(ENV, "TRY_CONTINUOUS", "false") == "true"

# Without a GPU, a GPU-type backend is still used to choose the GPU version of the callback
struct NoGPU <: KernelAbstractions.GPU end
const CALLBACK_BACKEND = USE_GPU ? BACKEND : NoGPU()

free_gpu_memory() = isdefined(CUDA, :free_memory) ? CUDA.free_memory() : CUDA.available_memory()   # CUDA.jl 6 / 5

println("="^88)
println("Device      : ", USE_GPU ? CUDA.name(CUDA.device()) : "no NVIDIA GPU available -> GPU kernel on the CPU")
USE_GPU && @printf("GPU memory  : %.2f GiB free of %.2f GiB\n", free_gpu_memory() / 2^30, CUDA.total_memory() / 2^30)
println("Versions    : Julia ", VERSION, " | DiffEqGPU ", pkgversion(DiffEqGPU), " | CUDA.jl ", pkgversion(CUDA),
        " | KernelAbstractions ", pkgversion(KernelAbstractions))
println("Threads     : ", Threads.nthreads(), " | rays per run: ", NRAYS)
println("="^88)

# ---------------- scenario: pulsar in Kerr-Schild coordinates, Float32 ----------------
const T         = Float32
const spacetime = Kerr.KerrSchildCoordinates(T(1), T(0.1))
const region    = PulsarRegion(R_in = T(5), R_out = T(50))
const tspan     = (zero(T), T(500))
const tol       = (abstol = T(1e-6), reltol = T(1e-6))

# Photons emitted between r = 8 and r = 15 in random directions, null momentum (g_{μν} k^μ k^ν = 0)
function pulsar_rays(n)
    rng  = Xoshiro(1234)
    data = zeros(T, 8, n)
    for j in 1:n
        d   = normalize3(randn(rng, T, 3))
        x   = SVector{4,T}(0, ((8 + 7 * rand(rng, T)) .* d)...)
        n3  = normalize3(randn(rng, T, 3))
        g   = metric(spacetime, x); ks = SVector(zero(T), n3...)
        a = g[1,1]; b = 2 * (g[1,2] * n3[1] + g[1,3] * n3[2] + g[1,4] * n3[3]); c = ks' * g * ks
        data[:, j] = vcat(x, SVector((-b - sqrt(b^2 - 4a*c)) / (2a), n3...))
    end
    return data
end
normalize3(v) = SVector{3,T}(v ./ sqrt(sum(abs2, v)))

function ensemble_problem(data)
    cfg = RT.Configuration(spacetime = spacetime, equations = RT.GeodesicEquationsVacuum(), backend = BACKEND,
                           u0 = SVector{8,T}(data[:, 1]), tspan = tspan)
    return RT.Ensemble_Problem(cfg, data)
end

# ---------------- callbacks ----------------
const cb_gpu        = boundary_callback(region, CALLBACK_BACKEND)        # the one of the library for the GPU
const cb_cpu        = boundary_callback(region, KernelAbstractions.CPU()) # the one of the library for the CPU
const cb_continuous = cb_cpu                                              # ContinuousCallback, only to show issue #219

solve_gpu(ep, n, cb) = solve(ep, GPUTsit5(), EnsembleGPUKernel(BACKEND); trajectories = n, callback = cb,
                             merge_callbacks = true, adaptive = true, save_everystep = false, save_start = false, tol...)
solve_cpu(ep, n, cb) = solve(ep, Tsit5(), EnsembleThreads(); trajectories = n, callback = cb,
                             save_everystep = false, save_start = false, tol...)

# ---------------- checks ----------------
radius(u) = coordinate_radius(spacetime, SVector(u[1], u[2], u[3], u[4]))
function classify(us, n)
    rs = [radius(us[i]) for i in 1:n]
    tol_border = 1f-3 * region.R_out
    hit     = count(r -> abs(r - region.R_in)  < tol_border, rs)
    escaped = count(r -> abs(r - region.R_out) < tol_border, rs)
    worst   = maximum(min(abs(r - region.R_in), abs(r - region.R_out)) for r in rs)
    return (hit = hit, escaped = escaped, other = n - hit - escaped, worst = worst)
end

methods = TRY_CONTINUOUS ? (("library GPU (Discrete)", cb_gpu), ("ContinuousCallback", cb_continuous)) :
                           (("library GPU (Discrete)", cb_gpu),)

# ---------------- 1) allocations inside the GPU kernel (cause of issue #219) ----------------
# Must run before any other solve: the kernel is inspected when it is compiled.
if USE_GPU
    println("\n1) Allocations inside the GPU kernel (gpu_malloc / gpu_gc_pool_alloc)")
    for (name, cb) in methods
        try
            ep = ensemble_problem(pulsar_rays(32))
            ir = sprint(io -> CUDA.@device_code_llvm io = io dump_module = true solve_gpu(ep, 32, cb))
            allocs = occursin("gpu_malloc", ir) || occursin("gpu_gc_pool_alloc", ir)
            println("   ", rpad(name, 24), allocs ? "ALLOCATES inside the kernel (problem of #219)" : "no allocations inside the kernel")
        catch e
            println("   ", rpad(name, 24), "could not be inspected: ", first(sprint(showerror, e), 200))
        end
    end
else
    println("\n1) Allocations inside the GPU kernel: skipped (needs a working NVIDIA GPU)")
end

# ---------------- 2) runs of increasing size ----------------
println("\n2) Runs with GPUTsit5 + EnsembleGPUKernel, Float32, tol = 1e-6")
@printf("   %-24s %10s %10s %10s %8s %14s %10s\n", "callback", "rays", "hit star", "escaped", "other", "max |r-border|", "time [s]")
largest = Dict{String,Any}()
for (name, cb) in methods
    for n in NRAYS
        try
            data = pulsar_rays(n); ep = ensemble_problem(data)
            solve_gpu(ep, min(n, 64), cb)                      # compilation
            t   = @elapsed (sol = solve_gpu(ep, n, cb))
            c   = classify(sol.u, n)
            @printf("   %-24s %10d %10d %10d %8d %14.2e %10.3f\n", name, n, c.hit, c.escaped, c.other, c.worst, t)
            largest[name] = (n = n, data = data, us = sol.u)
        catch e
            @printf("   %-24s %10d  ERROR: %s\n", name, n, first(sprint(showerror, e), 160))
            break
        end
    end
end

# ---------------- 3) comparison with the CPU (Tsit5 + ContinuousCallback, EnsembleThreads) ----------------
println("\n3) Comparison with the CPU (Tsit5 + boundary_callback(region, CPU())) on the first ", NCPU, " rays")
for (name, run) in largest
    n     = min(NCPU, run.n)
    ep    = ensemble_problem(run.data[:, 1:n])
    cpu   = solve_cpu(ep, n, cb_cpu)
    c_cpu = classify(cpu.u, n); c_gpu = classify(run.us, n)
    same  = count(i -> (radius(cpu.u[i]) < 10) == (radius(run.us[i]) < 10), 1:n)
    rel   = maximum(maximum(abs.(cpu.u[i] .- run.us[i]) ./ max.(1, abs.(cpu.u[i]))) for i in 1:n)
    @printf("   %-24s CPU hit/escaped %d/%d | GPU %d/%d | same border %d/%d | max rel. diff %.2e\n",
            name, c_cpu.hit, c_cpu.escaped, c_gpu.hit, c_gpu.escaped, same, n, rel)
end

println("\n", "="^88)
println("Copy this output and paste it in the conversation (or in dev/TAREAS.md, #16 part 1).")
println("="^88)
