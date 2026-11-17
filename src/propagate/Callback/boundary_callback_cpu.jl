# ======================================================================= #
# CPU: ContinuousCallback + terminate!                                    #
#   (Tsit5 / Vern7 / Vern9, EnsembleThreads)                              #
# The root finding of the ContinuousCallback places the state exactly on  #
# the boundary of the region. Only the final state is saved.              #
# The spacetime is read from the parameters p = IntegrationParameters.    #
# ======================================================================= #
function boundary_callback(region::AbstractRegion, ::KA.CPU)
    condition(u, t, integrator) = region_condition(region, integrator.p.spacetime, SVector(u[1], u[2], u[3], u[4]))
    return SciMLBase.ContinuousCallback(condition, SciMLBase.terminate!; save_positions = (false, false))
end
