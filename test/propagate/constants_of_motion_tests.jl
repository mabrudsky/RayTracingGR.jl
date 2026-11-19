using RayTracingGR
import Test: @testset, @test
import StaticArrays: SVector
import KernelAbstractions
import OrdinaryDiffEqTsit5: Tsit5

const RTM = RayTracingGR
# Uses pulsar_rays (callback_tests.jl)

# ======================================================================================== #
# Constants of motion of a photon in Kerr, written in Kerr-Schild coordinates (t, x, y, z) #
# Validates the whole chain: metric -> christoffel -> equations -> solver -> callback.     #
#   k_μ = g_{μν} k^ν                                                                       #
#   null condition  N   = g_{μν} k^μ k^ν                       (= 0)                       #
#   energy          E   = -k_t                                 (Killing vector ∂_t)         #
#   axial momentum  L_z = x k_y - y k_x                        (Killing vector ∂_φ)         #
#   Carter constant Q   = p_θ² + cos²θ (L_z² / sin²θ - a² E²)  (Killing tensor of Kerr)     #
# with cos θ = z / r (r: Kerr radius) and p_θ = k_μ ∂x^μ/∂θ. From                          #
#   x + i y = (r + i a) sin θ e^{iφ},  z = r cos θ                                          #
# at fixed (t, r, φ): ∂x/∂θ = x cot θ, ∂y/∂θ = y cot θ, ∂z/∂θ = -r sin θ, so               #
#   p_θ = cot θ (x k_x + y k_y) - r sin θ k_z                                               #
# Q is specific to Kerr: it lives here, not in the library.                                #
# ======================================================================================== #
function constants_of_motion(spacetime::Kerr.KerrSchildCoordinates, u)
    position = SVector(u[1], u[2], u[3], u[4])
    k        = SVector(u[5], u[6], u[7], u[8])
    g        = metric(spacetime, position)
    k_down   = g * k
    _, x, y, z = position

    r     = coordinate_radius(spacetime, position)
    cosθ  = z / r
    sinθ  = sqrt(1 - cosθ^2)
    E     = -k_down[1]
    L_z   = x * k_down[3] - y * k_down[2]
    p_θ   = (cosθ / sinθ) * (x * k_down[2] + y * k_down[3]) - r * sinθ * k_down[4]
    Q     = p_θ^2 + cosθ^2 * (L_z^2 / sinθ^2 - spacetime.a^2 * E^2)
    N     = (k' * g * k) / k[1]^2           # relative to (k^t)^2
    return (N = N, E = E, L_z = L_z, Q = Q)
end

# Change of a constant between the start and the end of each ray, relative to max(|initial value|, 1)
largest_change(f, initial, final) = maximum(abs(f(final[j]) - f(initial[j])) / max(abs(f(initial[j])), 1) for j in eachindex(initial))

# Tolerances: about 10 times the values measured on 2026-11-19 (a = 0.1 and 0.9, 200 rays)
#   Float64 (Vern9): N 5e-15, E 2e-15, L_z 1e-14, Q 1e-13
#   Float32 (Tsit5): N 2e-6,  E 1e-6,  L_z 3e-6,  Q 1e-4
@testset "Constants of motion along the rays ($T, a = $a)" for (T, alg) in ((Float64, nothing), (Float32, Tsit5())), a in (0.1, 0.9)
    spacetime = Kerr.KerrSchildCoordinates(T(1), T(a))
    region    = PulsarRegion(T(5), T(50))
    n         = 100
    data      = pulsar_rays(spacetime, n, T)
    cfg       = RTM.Configuration(spacetime = spacetime, equations = RTM.GeodesicEquationsVacuum(),
                                  backend = KernelAbstractions.CPU(), u0 = SVector{8}(data[:, 1]), tspan = (zero(T), T(500)))
    final     = alg === nothing ? propagate(cfg, data, region) : propagate(cfg, data, region; alg)   # Float64: Vern9 (default)

    initial_constants = [constants_of_motion(spacetime, data[:, j])  for j in 1:n]
    final_constants   = [constants_of_motion(spacetime, final[:, j]) for j in 1:n]

    tol = T == Float64 ? (N = 1e-13, E = 1e-13, L_z = 1e-13, Q = 1e-12) : (N = 2e-5, E = 1e-5, L_z = 3e-5, Q = 1e-3)
    @test maximum(abs(c.N) for c in final_constants) < tol.N
    @test largest_change(c -> c.E,   initial_constants, final_constants) < tol.E
    @test largest_change(c -> c.L_z, initial_constants, final_constants) < tol.L_z
    @test largest_change(c -> c.Q,   initial_constants, final_constants) < tol.Q
end
