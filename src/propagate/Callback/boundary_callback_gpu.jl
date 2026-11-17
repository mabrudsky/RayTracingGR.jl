# ======================================================================= #
# GPU: DiscreteCallback + bisection on the interpolation of the           #
# integrator + terminate!                                                 #
#   (GPUTsit5 / GPUVern7 / GPUVern9, EnsembleGPUKernel,                   #
#    merge_callbacks = true)                                              #
# ContinuousCallback makes the integrator escape to the GPU heap in       #
# EnsembleGPUKernel ("Out of dynamic GPU memory", DiffEqGPU issue #219).  #
# Every function that receives the integrator is @inline, so the          #
# integrator stays in registers.                                          #
# At the end of each step: if the ray left the region, the crossing time  #
# is found by bisection between tprev (inside) and t (outside) using the  #
# interpolation integrator(t), the state is moved there and the           #
# integration stops. Only the final state is saved.                       #
# The spacetime is read from the parameters p = IntegrationParameters.    #
# ======================================================================= #
function boundary_callback(region::AbstractRegion, ::KA.GPU)
    @inline inside(u, integrator) = region_condition(region, integrator.p.spacetime, SVector(u[1], u[2], u[3], u[4]))
    @inline condition(u, t, integrator) = inside(u, integrator) <= 0
    @inline function affect!(integrator)
        t_in, t_out = integrator.tprev, integrator.t
        for _ in 1:50
            t_mid = (t_in + t_out) / 2
            inside(integrator(t_mid), integrator) > 0 ? (t_in = t_mid) : (t_out = t_mid)
        end
        SciMLBase.change_t_via_interpolation!(integrator, t_out)
        SciMLBase.terminate!(integrator)
    end
    return SciMLBase.DiscreteCallback(condition, affect!; save_positions = (false, false))
end
