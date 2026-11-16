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

# ---------------- scenario: pulsar in Kerr-Schild coordinates, Float32 ----------------
const T         = Float32
const spacetime = Kerr.KerrSchildCoordinates(T(1), T(0.1))
const region    = PulsarRegion(R_in = T(5), R_out = T(50))
const tspan     = (zero(T), T(500))
const tol       = (abstol = T(1e-6), reltol = T(1e-6))

const LINE = "="^96
println(LINE)
println("TEST: stopping the rays at the boundary of the region (boundary_callback) on the GPU")
println(LINE)
println("Scenario     : pulsar, Kerr spacetime in Kerr-Schild coordinates (M = 1, a = 0.1).")
println("               Photons are emitted between r = 8 and r = 15 in random directions and are")
println("               integrated until they reach the star surface (r = ", region.R_in, ") or the outer radius (r = ", region.R_out, ").")
println("               Precision Float32, tolerance 1e-6, integrator GPUTsit5.")
println("Device       : ", USE_GPU ? CUDA.name(CUDA.device()) : "no NVIDIA GPU available -> the GPU code runs on the CPU")
USE_GPU && @printf("GPU memory   : %.2f GiB free of %.2f GiB\n", free_gpu_memory() / 2^30, CUDA.total_memory() / 2^30)
println("Software     : Julia ", VERSION, ", DiffEqGPU ", pkgversion(DiffEqGPU), ", CUDA.jl ", pkgversion(CUDA),
        ", KernelAbstractions ", pkgversion(KernelAbstractions))
println("CPU threads  : ", Threads.nthreads(), " (used only for the comparison with the CPU)")

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
verdict(ok) = ok ? "OK" : "FAILED"

methods = TRY_CONTINUOUS ? (("GPU callback of the library", cb_gpu), ("ContinuousCallback (optional)", cb_continuous)) :
                           (("GPU callback of the library", cb_gpu),)
all_ok   = true
failures = String[]

# ---------------- 1) memory requested by the GPU code while it runs (cause of issue #219) ----------------
# Must run before any other solve: the GPU code is inspected when it is compiled.
println("\n", LINE)
println("[1] Does the GPU code request memory while it runs?")
println("    If it does, many rays at once fill the GPU memory and the run stops with")
println("    \"Out of dynamic GPU memory\". The right answer is NO.")
println(LINE)
if USE_GPU
    for (name, cb) in methods
        try
            ep = ensemble_problem(pulsar_rays(32))
            ir = sprint(io -> CUDA.@device_code_llvm io = io dump_module = true solve_gpu(ep, 32, cb))
            allocs = occursin("gpu_malloc", ir) || occursin("gpu_gc_pool_alloc", ir)
            @printf("    %-32s %-5s -> %s\n", name, allocs ? "YES" : "NO", verdict(!allocs))
            if allocs && name == "GPU callback of the library"
                global all_ok = false; push!(failures, "the GPU code of the library requests memory while it runs")
            end
        catch e
            @printf("    %-32s could not be checked: %s\n", name, first(sprint(showerror, e), 160))
        end
    end
else
    println("    Skipped: it needs a working NVIDIA GPU.")
end

# ---------------- 2) runs of increasing size ----------------
println("\n", LINE)
println("[2] Integration on the GPU with more and more rays")
println("    Each ray must stop exactly on one of the two boundaries:")
println("      - hit the star   : stopped at the star surface (r = ", region.R_in, ")")
println("      - escaped        : stopped at the outer radius (r = ", region.R_out, ")")
println("      - not stopped    : did not reach any boundary. Must be 0.")
println("      - largest error  : largest distance between a ray and its boundary (Float32: about 1e-5 to 1e-4)")
println("      - time           : time of the GPU integration, in seconds")
println(LINE)
@printf("    %-32s %10s %13s %10s %12s %14s %9s  %s\n", "callback", "rays", "hit the star", "escaped",
        "not stopped", "largest error", "time [s]", "result")
largest = Dict{String,Any}()
for (name, cb) in methods
    for n in NRAYS
        try
            data = pulsar_rays(n); ep = ensemble_problem(data)
            solve_gpu(ep, min(n, 64), cb)                      # compilation
            t   = @elapsed (sol = solve_gpu(ep, n, cb))
            c   = classify(sol.u, n)
            ok  = c.other == 0 && c.worst < 1f-3 * region.R_out
            @printf("    %-32s %10d %13d %10d %12d %14.1e %9.3f  %s\n", name, n, c.hit, c.escaped, c.other, c.worst, t, verdict(ok))
            largest[name] = (n = n, data = data, us = sol.u)
            if !ok && name == "GPU callback of the library"
                global all_ok = false; push!(failures, "$(c.other) of $n rays did not stop at a boundary")
            end
        catch e
            @printf("    %-32s %10d  ERROR: %s  %s\n", name, n, first(sprint(showerror, e), 120), verdict(false))
            if name == "GPU callback of the library"
                global all_ok = false; push!(failures, "the GPU run with $n rays stopped with an error")
            end
            break
        end
    end
end

# ---------------- 3) comparison with the CPU ----------------
println("\n", LINE)
println("[3] Do the GPU and the CPU give the same result?")
println("    The first rays are integrated again on the CPU (reference: Tsit5 + ContinuousCallback)")
println("    and compared ray by ray with the GPU.")
println(LINE)
for (name, _) in methods
    haskey(largest, name) || continue
    run   = largest[name]
    n     = min(NCPU, run.n)
    ep    = ensemble_problem(run.data[:, 1:n])
    cpu   = solve_cpu(ep, n, cb_cpu)
    c_cpu = classify(cpu.u, n); c_gpu = classify(run.us, n)
    same  = count(i -> (radius(cpu.u[i]) < 10) == (radius(run.us[i]) < 10), 1:n)
    rel   = maximum(maximum(abs.(cpu.u[i] .- run.us[i]) ./ max.(1, abs.(cpu.u[i]))) for i in 1:n)
    ok    = same == n && rel < 1e-2
    println("    ", name, " (", n, " rays):")
    @printf("      rays that hit the star          : CPU %d, GPU %d\n", c_cpu.hit, c_gpu.hit)
    @printf("      rays that escaped               : CPU %d, GPU %d\n", c_cpu.escaped, c_gpu.escaped)
    @printf("      rays that end on the same boundary on CPU and GPU : %d of %d\n", same, n)
    @printf("      largest relative difference of the final state   : %.1e  (Float32: about 1e-5 to 1e-3)\n", rel)
    println("      result: ", verdict(ok))
    if !ok && name == "GPU callback of the library"
        global all_ok = false; push!(failures, "the GPU and the CPU do not give the same result")
    end
end

println("\n", LINE)
if all_ok && USE_GPU
    println("SUMMARY: OK. The rays stop at the boundary on the GPU, without memory errors,")
    println("         and the GPU gives the same result as the CPU.")
elseif all_ok
    println("SUMMARY: OK, but without a GPU: the GPU code ran on the CPU and the memory check [1]")
    println("         was skipped. Run this test on a machine with an NVIDIA GPU.")
else
    println("SUMMARY: FAILED.")
    foreach(f -> println("         - ", f), failures)
end
println(LINE)
