# ======================================================================= #
# GPU: DiscreteCallback + bisection on the interpolation of the           #
# integrator + terminate!                                                 #
#   (GPUTsit5 / GPUVern7 / GPUVern9, EnsembleGPUKernel,                   #
#    merge_callbacks = true)                                              #
# ContinuousCallback makes the integrator escape to the GPU heap in       #
# EnsembleGPUKernel ("Out of dynamic GPU memory", DiffEqGPU issue #219).  #
# Every function that receives the integrator is @inline, so the          #
# integrator stays in registers.                                          #
# At the end of each step: if region_condition changed sign between the   #
# previous state (uprev) and the current one (u), the crossing time is    #
# found by bisection between tprev and t using the interpolation          #
# integrator(t). If stops_at_boundary says so, the state is moved there   #
# and the integration stops; if not (e.g. the equatorial plane outside an #
# accretion disk), the integrator is not touched and the ray goes on.     #
# Only the final state is saved.                                          #
# After the bisection, one Newton step along the ray puts the state on    #
# the boundary c = 0 (in Float32 the bisection alone leaves it up to      #
# ~9 ulp away; with the Newton step, 1-2 ulp):                            #
#   Δλ = -c(x) / (dc/dλ),   dc/dλ = ∂_μ c k^μ  (ForwardDiff dual)         #
#   u  ← u + Δλ f(u)        (error ~ Δλ² ~ 1e-12, negligible)             #
# The spacetime is read from the parameters p = IntegrationParameters.    #
# ======================================================================= #
function boundary_callback(region::AbstractRegion, ::KA.GPU)
    @inline position(u) = SVector(u[1], u[2], u[3], u[4])
    @inline c(u, integrator) = region_condition(region, integrator.p.spacetime, position(u))
    # Sign change of the condition between the previous and the current state
    @inline condition(u, t, integrator) = (c(integrator.uprev, integrator) > 0) != (c(u, integrator) > 0)
    # One Newton step along the ray: moves the state (position and momentum) until c = 0
    @inline function newton_to_boundary!(integrator)
        u, p      = integrator.u, integrator.p
        x         = position(u)
        k         = SVector(u[5], u[6], u[7], u[8])
        c_x       = region_condition(region, p.spacetime, x)
        dc_dλ     = derivative(s -> region_condition(region, p.spacetime, x + s * k), zero(eltype(u)))
        Δλ        = -c_x / dc_dλ
        integrator.u = u + Δλ * integrator.f(u, p, integrator.t)
    end
    @inline function affect!(integrator)
        side_before   = c(integrator.uprev, integrator) > 0   # side of the surface at tprev
        t_in, t_out   = integrator.tprev, integrator.t
        for _ in 1:50
            t_mid = (t_in + t_out) / 2
            (c(integrator(t_mid), integrator) > 0) == side_before ? (t_in = t_mid) : (t_out = t_mid)
        end
        if stops_at_boundary(region, integrator.p.spacetime, position(integrator(t_out)))
            SciMLBase.change_t_via_interpolation!(integrator, t_out)
            newton_to_boundary!(integrator)
            SciMLBase.terminate!(integrator)
        end
    end
    return SciMLBase.DiscreteCallback(condition, affect!; save_positions = (false, false))
end
