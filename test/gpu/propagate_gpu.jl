# ==================================================================================== #
# Test of propagate on a real NVIDIA GPU. Not part of Pkg.test() (the CI has no GPU).  #
# Run (the script activates its own environment, test/gpu/, from any folder):          #
#   julia -t auto test/gpu/propagate_gpu.jl          # from the project folder         #
#   julia -t auto propagate_gpu.jl                   # from test/gpu/                  #
# The first time, Pkg.instantiate() installs the packages of test/gpu/Project.toml.    #
# Options (environment variables):                                                     #
#   NRAYS = "10000,100000,1000000"   number of rays of each run                        #
#   NCPU  = "10000"                  rays compared against the CPU (Float64)           #
# Checks:                                                                              #
#   [1] the GPU code does not request memory while it runs                             #
#   [2] runs of increasing size: every ray stops at most 3 ulp from its boundary       #
#   [3] the GPU (Float32) gives the same result as the CPU (Float64)                   #
# ==================================================================================== #
import Pkg
Pkg.activate(@__DIR__)    # environment of test/gpu/, wherever the script is called from
Pkg.instantiate()         # installs what is missing (only the first time)

using RayTracingGR, StaticArrays, KernelAbstractions, CUDA, DiffEqGPU
using Random, Printf
const RT = RayTracingGR

const LINE = "="^100
if !CUDA.functional()
    println(LINE, "\nNo working NVIDIA GPU found: this test needs one. Nothing was run.\n", LINE)
    exit(1)
end

const NRAYS = parse.(Int, split(get(ENV, "NRAYS", "10000,100000,1000000"), ","))
const NCPU  = parse(Int, get(ENV, "NCPU", "10000"))
free_gpu_memory() = isdefined(CUDA, :free_memory) ? CUDA.free_memory() : CUDA.available_memory()   # CUDA.jl 6 / 5

# ---------------- scenario: pulsar in Kerr-Schild coordinates ----------------
const spacetime   = Kerr.KerrSchildCoordinates(1f0, 0.1f0)   # GPU: Float32
const region      = PulsarRegion(5f0, 50f0)
const spacetime64 = Kerr.KerrSchildCoordinates(1.0, 0.1)     # CPU: Float64 (reference)
const region64    = PulsarRegion(5.0, 50.0)

configuration(st, data, backend) =
    RT.Configuration(spacetime = st, equations = RT.GeodesicEquationsVacuum(), backend = backend,
                     u0 = SVector{8}(data[:, 1]), tspan = (zero(eltype(data)), eltype(data)(500)))

# Photons emitted between r = 8 and r = 15 in random directions, null momentum (g_{μν} k^μ k^ν = 0)
function pulsar_rays(n)
    rng  = Xoshiro(1234)
    data = zeros(Float32, 8, n)
    for j in 1:n
        d  = unit3(randn(rng, Float32, 3))
        x  = SVector{4,Float32}(0, ((8 + 7 * rand(rng, Float32)) .* d)...)
        n3 = unit3(randn(rng, Float32, 3))
        g  = metric(spacetime, x); ks = SVector(0f0, n3...)
        a = g[1,1]; b = 2 * (g[1,2] * n3[1] + g[1,3] * n3[2] + g[1,4] * n3[3]); c = ks' * g * ks
        data[:, j] = vcat(x, SVector((-b - sqrt(b^2 - 4a*c)) / (2a), n3...))
    end
    return data
end
unit3(v) = SVector{3,Float32}(v ./ sqrt(sum(abs2, v)))

# ---------------- measurements on the final states (one column per ray) ----------------
radius(st, u)   = coordinate_radius(st, SVector(u[1], u[2], u[3], u[4]))
ulps_to_border(r) = min(abs(r - region.R_in) / eps(region.R_in), abs(r - region.R_out) / eps(region.R_out))
quantile_of(v, p) = sort(v)[clamp(ceil(Int, p * length(v)), 1, length(v))]
verdict(ok) = ok ? "OK" : "FAILED"

all_ok   = true
failures = String[]
fail!(msg) = (global all_ok = false; push!(failures, msg))

println(LINE)
println("TEST: propagate on the GPU")
println(LINE)
println("Scenario   : pulsar, Kerr in Kerr-Schild coordinates (M = 1, a = 0.1). Photons emitted between")
println("             r = 8 and r = 15, integrated until r = ", region.R_in, " (star) or r = ", region.R_out, " (escape).")
println("Call       : propagate(configuration, initial_data, region) with backend = CUDA.CUDABackend(), Float32")
println("Defaults   : ", nameof(typeof(RT.default_algorithm(CUDA.CUDABackend()))), ", abstol = reltol = 5 eps(Float32) = ",
        RT.default_tolerance(Float32))
println("Device     : ", CUDA.name(CUDA.device()))
@printf("GPU memory : %.2f GiB free of %.2f GiB\n", free_gpu_memory() / 2^30, CUDA.total_memory() / 2^30)
println("Software   : Julia ", VERSION, ", DiffEqGPU ", pkgversion(DiffEqGPU), ", CUDA.jl ", pkgversion(CUDA))
@printf("ulp        : ulp(5) = %.1e, ulp(50) = %.1e  (smallest possible distance to each boundary in Float32)\n",
        eps(region.R_in), eps(region.R_out))

# ---------------- [1] memory requested by the GPU code while it runs ----------------
# Must run before any other propagate: the GPU code is inspected when it is compiled.
println("\n", LINE)
println("[1] Does the GPU code request memory while it runs?")
println("    If it does, many rays fill the GPU memory (\"Out of dynamic GPU memory\"). Expected: NO.")
println(LINE)
let data = pulsar_rays(32)
    ir = sprint(io -> CUDA.@device_code_llvm io = io dump_module = true propagate(configuration(spacetime, data, CUDA.CUDABackend()), data, region))
    allocs = occursin("gpu_malloc", ir) || occursin("gpu_gc_pool_alloc", ir)
    println("    memory requested inside the kernel: ", allocs ? "YES" : "NO", "  -> ", verdict(!allocs))
    allocs && fail!("the GPU code requests memory while it runs")
end

# ---------------- [2] runs of increasing size ----------------
println("\n", LINE)
println("[2] propagate with more and more rays")
println("    star / escaped : rays stopped at r = ", region.R_in, " / r = ", region.R_out)
println("    not stopped    : rays that did not reach a boundary. Expected: 0.")
println("    ulp p99 / max  : distance to the boundary in ulp (99% of the rays / worst ray). Expected: <= 3.")
println("    time           : whole call to propagate, in seconds")
println(LINE)
@printf("    %10s %8s %10s %12s %9s %9s %10s  %s\n", "rays", "star", "escaped", "not stopped", "ulp p99", "ulp max", "time [s]", "result")
largest = nothing
for n in NRAYS
    data = pulsar_rays(n)
    cfg  = configuration(spacetime, data, CUDA.CUDABackend())
    propagate(cfg, data[:, 1:min(n, 64)], region)               # compilation
    t     = @elapsed (final = propagate(cfg, data, region))
    rs    = [radius(spacetime, final[:, j]) for j in 1:n]
    ulps  = ulps_to_border.(rs)
    star  = count(r -> abs(r - region.R_in)  < 0.05f0, rs)
    esc   = count(r -> abs(r - region.R_out) < 0.05f0, rs)
    other = n - star - esc
    ok    = other == 0 && maximum(ulps) <= 3
    @printf("    %10d %8d %10d %12d %9.1f %9.1f %10.3f  %s\n", n, star, esc, other, quantile_of(ulps, 0.99), maximum(ulps), t, verdict(ok))
    other > 0         && fail!("$other of $n rays did not stop at a boundary")
    maximum(ulps) > 3 && fail!(@sprintf("with %d rays the worst ray stopped %.1f ulp from its boundary", n, maximum(ulps)))
    global largest = (data = data, final = final)
end

# ---------------- [3] comparison with the CPU in Float64 ----------------
n = min(NCPU, size(largest.data, 2))
println("\n", LINE)
println("[3] Does the GPU (Float32) give the same result as the CPU (Float64)?")
println("    The first ", n, " rays again with propagate on the CPU (Vern9, Float64), same initial data.")
println(LINE)
data64 = Float64.(largest.data[:, 1:n])
cfg64  = configuration(spacetime64, data64, KernelAbstractions.CPU())
propagate(cfg64, data64[:, 1:16], region64)                     # compilation
t_cpu  = @elapsed (cpu = propagate(cfg64, data64, region64))
gpu    = Float64.(largest.final[:, 1:n])
same   = count(j -> (radius(spacetime64, cpu[:, j]) < 10) == (radius(spacetime64, gpu[:, j]) < 10), 1:n)
rel    = [maximum(abs.(gpu[:, j] .- cpu[:, j]) ./ max.(abs.(cpu[:, j]), 1)) for j in 1:n]
big    = count(>(1e-3), rel)
@printf("    rays on the same boundary on CPU and GPU    : %d of %d\n", same, n)
@printf("    relative difference of the final state      : median %.1e, 99%% %.1e  (expected ~3e-7 and ~1e-5)\n",
        quantile_of(rel, 0.5), quantile_of(rel, 0.99))
@printf("    rays with relative difference > 1e-3        : %d  (rays near the critical orbit; a few per 5000)\n", big)
@printf("    CPU time                                    : %.2f s\n", t_cpu)
ok = big <= max(1, n ÷ 500) && quantile_of(rel, 0.5) < 1e-5
println("    result: ", verdict(ok))
ok || fail!("the GPU and the CPU do not give the same result")

# ---------------- summary ----------------
println("\n", LINE)
if all_ok
    println("SUMMARY: OK. propagate works on the GPU without requesting memory inside the kernel, every ray")
    println("         stops at most 3 ulp from its boundary, and the GPU agrees with the CPU in Float64.")
else
    println("SUMMARY: FAILED.")
    foreach(f -> println("         - ", f), failures)
end
println(LINE)
