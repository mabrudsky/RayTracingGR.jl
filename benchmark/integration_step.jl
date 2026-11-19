# ==================================================================================== #
# Cost of ONE integration step on the CPU (Float64): given a state u_n, how long does  #
# it take to obtain the next state u_{n+1}? Measured with BenchmarkTools.              #
# Not part of the tests. Run from the project folder:                                  #
#   julia --project=benchmark benchmark/integration_step.jl                            #
# Prints a table and saves benchmark/integration_step.png (used in the README).        #
#                                                                                      #
# How:                                                                                 #
#   1. A real trajectory of the pulsar is integrated, saving every state u_n and the   #
#      step size h_n used by the simulation.                                           #
#   2. For several states along it (from the emission to the boundary), an integrator  #
#      is created at u_n with h_n and the same options as propagate on the CPU:        #
#      Vern9, tolerance 5 eps(Float64), boundary callback.                             #
#   3. @benchmark repeats only step!(integrator), u_n -> u_{n+1}, many times from the   #
#      same u_n (creating the integrator again before each repetition is not timed).   #
# The step includes everything the simulation does between two states: the 16         #
# evaluations of the right-hand side f(u), the combination of the stages, the error   #
# control and the check of the boundary callback.                                      #
# ==================================================================================== #
using RayTracingGR, StaticArrays, KernelAbstractions, SciMLBase, BenchmarkTools, Printf, Statistics
import OrdinaryDiffEqVerner: Vern9
using CairoMakie
const RT = RayTracingGR

# ---------------- scenario: pulsar in Kerr-Schild coordinates, Float64 ----------------
const spacetime = Kerr.KerrSchildCoordinates(1.0, 0.9)
const region    = PulsarRegion(5.0, 50.0)
const p         = RT.IntegrationParameters(spacetime)
const tol       = RT.default_tolerance(Float64)
const callback  = boundary_callback(region, CPU())
const NSTATES   = 7                        # states measured along the trajectory

# Photon emitted at r ≈ 11 with null momentum (g_{μν} k^μ k^ν = 0).
# Parallel transport: the transported vector f^μ = (0, 1, 0, 0) is appended (only for timing).
function initial_state(N)
    x  = SVector(0.0, 8.0, 5.0, 6.0)
    n3 = SVector(0.6, 0.0, 0.8)
    g  = metric(spacetime, x); ks = SVector(0.0, n3...)
    a = g[1,1]; b = 2 * (g[1,2] * n3[1] + g[1,3] * n3[2] + g[1,4] * n3[3]); c = ks' * g * ks
    u = vcat(x, SVector((-b - sqrt(b^2 - 4a*c)) / (2a), n3...))
    return N == 8 ? u : vcat(u, SVector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
end

new_integrator(prob, h) = SciMLBase.init(prob, Vern9(); callback = callback, abstol = tol, reltol = tol, dt = h,
                                         save_everystep = false, save_start = false)

function measure(name, rhs, N)
    # 1. real trajectory: every state u_n and step size h_n of the simulation
    prob    = SciMLBase.ODEProblem{false}(rhs, initial_state(N), (0.0, 500.0), p)
    traj    = SciMLBase.solve(prob, Vern9(); callback = callback, abstol = tol, reltol = tol)
    n_steps = length(traj.t) - 1
    # 2. and 3. one step u_n -> u_{n+1}, repeated many times from the same u_n
    states = map(round.(Int, range(2, n_steps - 2, length = NSTATES))) do n
        u_n, h_n = traj.u[n], traj.t[n+1] - traj.t[n]
        prob_n   = SciMLBase.remake(prob; u0 = u_n, tspan = (traj.t[n], 500.0))
        t        = median(@benchmark SciMLBase.step!(integ) setup = (integ = new_integrator($prob_n, $h_n)) evals = 1).time / 1000
        r        = coordinate_radius(spacetime, SVector(u_n[1], u_n[2], u_n[3], u_n[4]))
        (n = n, r = r, h = h_n, t = t)
    end
    return (name = name, n_steps = n_steps, states = states, t_step = median([s.t for s in states]))
end

println("CPU    : ", Sys.cpu_info()[1].model)
println("Julia  : ", VERSION, ", Float64, Vern9, tolerance 5 eps = ", tol)
println("Case   : Kerr-Schild a = 0.9, pulsar 5 < r < 50, one ray, ", NSTATES, " states along it")

results = [measure("geodésica", RT.geodesic_equations_Vacuum, 8),
           measure("transporte paralelo", RT.parallel_transport_equations_Vacuum, 16)]

for res in results
    println("\n--- ", res.name, " (trajectory of ", res.n_steps, " steps) ---")
    @printf("    %6s %8s %8s %16s\n", "step n", "r", "h_n", "u_n -> u_{n+1}")
    for s in res.states
        @printf("    %6d %8.1f %8.3f %13.2f µs\n", s.n, s.r, s.h, s.t)
    end
    @printf("    median: %.2f µs per step -> whole trajectory ≈ %d × %.2f µs ≈ %.2f ms\n",
            res.t_step, res.n_steps, res.t_step, res.n_steps * res.t_step / 1000)
end

# ---------------- figure ----------------
names = [res.name for res in results]
fig   = Figure(size = (900, 400))
ax    = Axis(fig[1, 1]; xlabel = "tiempo de un paso [µs]", yticks = (1:2, names), yreversed = true,
             title = "Costo de un paso de integración uₙ → uₙ₊₁ en CPU (Float64)",
             subtitle = "Vern9, tolerancia 5ε, incluye control del error y chequeo del borde\n" *
                        "Púlsar en Kerr-Schild (a = 0.9), $(Sys.cpu_info()[1].model)")
barplot!(ax, 1:2, [res.t_step for res in results]; direction = :x, color = [:steelblue, :darkorange])
for (i, res) in enumerate(results)
    # one point per measured state: the cost does not depend on where the ray is
    points = [s.t for s in res.states]
    scatter!(ax, points, fill(i, length(points)); color = :black, markersize = 7)
    text!(ax, maximum(points), i; text = @sprintf("   %.1f µs por paso", res.t_step), align = (:left, :center), fontsize = 14)
end
xlims!(ax, 0, 1.7 * maximum(res.t_step for res in results))
Label(fig[2, 1], "Barra: mediana. Puntos: cada uno de los $NSTATES estados medidos, desde la emisión hasta el borde.";
      fontsize = 12)
path = joinpath(@__DIR__, "integration_step.png")
save(path, fig; px_per_unit = 2)
println("\nFigure saved to ", path)
