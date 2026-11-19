# ==================================================================================== #
# Test of propagate on a real GPU. Not part of Pkg.test() (the CI has no GPU).         #
# The GPU is chosen with the environment variable GPU (default: metal on macOS, cuda   #
# on Linux / Windows). Each GPU has its own environment and backend file:              #
#   GPU=cuda   : NVIDIA, CUDA.jl      environment test/gpu/         backend_cuda.jl    #
#   GPU=amdgpu : AMD, AMDGPU.jl       environment test/gpu/amdgpu/  backend_amdgpu.jl  #
#   GPU=oneapi : Intel, oneAPI.jl     environment test/gpu/oneapi/  backend_oneapi.jl  #
#   GPU=metal  : Apple, Metal.jl      environment test/gpu/metal/   backend_metal.jl   #
# Only CUDA has been tested on a real GPU (A30); the others are written but untested.  #
# Run (the script activates the right environment, from any folder):                   #
#   julia -t auto test/gpu/propagate_gpu.jl              # default GPU of this system   #
#   GPU=amdgpu julia -t auto test/gpu/propagate_gpu.jl   # another GPU                  #
# The first time, Pkg.instantiate() installs the packages of the environment.          #
# Options (environment variables):                                                     #
#   GPU   = "cuda" | "amdgpu" | "oneapi" | "metal"                                     #
#   NRAYS = "10000,100000,1000000"   number of rays of each run                        #
#   NCPU  = "10000"                  rays compared against the CPU (Float64)           #
# Checks:                                                                              #
#   [1] the GPU code does not request memory while it runs                             #
#   [2] runs of increasing size: every ray stops at most 3 ulp from its boundary       #
#   [3] the GPU (Float32) gives the same result as the CPU (Float64)                   #
#   [4] accretion disk (AccretionDiskRegion): memory, where the rays end, vs CPU       #
# ==================================================================================== #
const GPU_KIND = lowercase(get(ENV, "GPU", Sys.isapple() ? "metal" : "cuda"))
GPU_KIND in ("cuda", "amdgpu", "oneapi", "metal") ||
    error("GPU = \"$GPU_KIND\" is not supported; use cuda, amdgpu, oneapi or metal")

import Pkg
Pkg.activate(GPU_KIND == "cuda" ? (@__DIR__) : joinpath(@__DIR__, GPU_KIND))   # environment of the chosen GPU
Pkg.instantiate()                                                           # installs what is missing (only the first time)

using RayTracingGR, StaticArrays, KernelAbstractions, DiffEqGPU
using Random, Printf
const RT = RayTracingGR
include(joinpath(@__DIR__, "backend_$(GPU_KIND).jl"))   # gpu_backend(), gpu_functional(), gpu_device_name(), ...

const LINE = "="^100
if !gpu_functional()
    println(LINE, "\nNo working GPU found ($(GPU_NAME)): this test needs one. Nothing was run.\n", LINE)
    exit(1)
end

const NRAYS = parse.(Int, split(get(ENV, "NRAYS", "10000,100000,1000000"), ","))
const NCPU  = parse(Int, get(ENV, "NCPU", "10000"))

# Memory check of the GPU code: prints "could not be checked" if the backend does not support it
function check_kernel_memory(label, f)
    try
        allocs = kernel_allocates(f)
        println("    memory requested inside the kernel: ", allocs ? "YES" : "NO", "  -> ", verdict(!allocs))
        allocs && fail!("the GPU code $label requests memory while it runs")
    catch e
        println("    could not be checked on this GPU: ", first(sprint(showerror, e), 200))
    end
end

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
println("Call       : propagate(configuration, initial_data, region) with backend = ", nameof(typeof(gpu_backend())), "(), Float32")
println("Defaults   : ", nameof(typeof(RT.default_algorithm(gpu_backend()))), ", abstol = reltol = 5 eps(Float32) = ",
        RT.default_tolerance(Float32))
println("GPU        : ", GPU_NAME)
println("Device     : ", try gpu_device_name() catch; "(unknown)" end)
println("GPU memory : ", try gpu_memory_text() catch; "(unknown)" end)
println("Float64    : ", KernelAbstractions.supports_float64(gpu_backend()) ? "supported" : "NOT supported by this GPU (only Float32)")
println("Software   : Julia ", VERSION, ", DiffEqGPU ", pkgversion(DiffEqGPU), ", KernelAbstractions ", pkgversion(KernelAbstractions))
@printf("ulp        : ulp(5) = %.1e, ulp(50) = %.1e  (smallest possible distance to each boundary in Float32)\n",
        eps(region.R_in), eps(region.R_out))

# ---------------- [1] memory requested by the GPU code while it runs ----------------
# Must run before any other propagate: the GPU code is inspected when it is compiled.
println("\n", LINE)
println("[1] Does the GPU code request memory while it runs?")
println("    If it does, many rays fill the GPU memory (\"Out of dynamic GPU memory\"). Expected: NO.")
println(LINE)
let data = pulsar_rays(32)
    check_kernel_memory("(pulsar)", () -> propagate(configuration(spacetime, data, gpu_backend()), data, region))
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
    cfg  = configuration(spacetime, data, gpu_backend())
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

# ---------------- [4] accretion disk ----------------
# Same rays, now with an accretion disk: the ray stops on the equatorial plane only inside the disk
# (6 ≤ r ≤ 15), at R_min = 2.05 (falls) or at R_max = 50 (escapes). Crossing the plane outside the
# disk must NOT stop it. The GPU callback is the same (sign change + bisection + Newton step).
const disk   = AccretionDiskRegion(R_min = 2.05f0, R_max = 50f0, R_in = 6f0, R_out = 15f0)
const disk64 = AccretionDiskRegion(R_min = 2.05, R_max = 50.0, R_in = 6.0, R_out = 15.0)
where_it_ends(st, region, u) = begin
    r = radius(st, u); z = coordinate_z(st, SVector(u[1], u[2], u[3], u[4]))
    abs(r - region.R_min) < 1e-3 * region.R_max ? :fell :
    abs(r - region.R_max) < 1e-3 * region.R_max ? :escaped :
    (abs(z) < 1e-3 * region.R_max && region.R_in <= r <= region.R_out) ? :disk : :other
end
println("\n", LINE)
println("[4] Accretion disk (AccretionDiskRegion): stops on the disk, at R_min or at R_max")
println("    Expected: no memory inside the kernel, 0 rays ending elsewhere, same ending as the CPU (Float64).")
println(LINE)
let data = pulsar_rays(32)
    check_kernel_memory("(disk)", () -> propagate(configuration(spacetime, data, gpu_backend()), data, disk))
end
let n = min(NCPU, size(largest.data, 2)), data = largest.data[:, 1:n]
    cfg = configuration(spacetime, data, gpu_backend())
    propagate(cfg, data[:, 1:64], disk)                                    # compilation
    t      = @elapsed (final = propagate(cfg, data, disk))
    ends   = [where_it_ends(spacetime, disk, final[:, j]) for j in 1:n]
    data64 = Float64.(data); cfg64 = configuration(spacetime64, data64, KernelAbstractions.CPU())
    cpu    = propagate(cfg64, data64, disk64)
    ends64 = [where_it_ends(spacetime64, disk64, cpu[:, j]) for j in 1:n]
    same   = count(ends .== ends64)
    @printf("    rays: %d | disk %d, fell %d, escaped %d, elsewhere %d | GPU time %.3f s\n", n,
            count(==(:disk), ends), count(==(:fell), ends), count(==(:escaped), ends), count(==(:other), ends), t)
    @printf("    same ending on GPU (Float32) and CPU (Float64): %d of %d\n", same, n)
    ok = count(==(:other), ends) == 0 && same >= n - max(1, n ÷ 500)
    println("    result: ", verdict(ok))
    ok || fail!("the disk on the GPU does not stop the rays where expected or differs from the CPU")
end

# ---------------- summary ----------------
println("\n", LINE)
if all_ok
    println("SUMMARY: OK. propagate works on the GPU without requesting memory inside the kernel, every ray")
    println("         stops at most 3 ulp from its boundary, the GPU agrees with the CPU in Float64, and the")
    println("         accretion disk stops the rays where expected.")
else
    println("SUMMARY: FAILED.")
    foreach(f -> println("         - ", f), failures)
end
println(LINE)
