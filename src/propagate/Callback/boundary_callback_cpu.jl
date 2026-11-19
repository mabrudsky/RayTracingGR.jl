# ======================================================================= #
# CPU: ContinuousCallback + terminate!                                    #
#   (Tsit5 / Vern7 / Vern9, EnsembleThreads)                              #
# The root finding of the ContinuousCallback places the state exactly on  #
# the surface where region_condition changes sign. There the ray stops    #
# if stops_at_boundary says so (always for PulsarRegion; for the disk,    #
# not when it crosses the equatorial plane outside the disk).             #
# Only the final state is saved.                                          #
# The spacetime is read from the parameters p = IntegrationParameters.    #
# ======================================================================= #
function boundary_callback(region::AbstractRegion, ::KA.CPU)
    position(u) = SVector(u[1], u[2], u[3], u[4])
    condition(u, t, integrator) = region_condition(region, integrator.p.spacetime, position(u))
    function affect!(integrator)
        stops_at_boundary(region, integrator.p.spacetime, position(integrator.u)) && SciMLBase.terminate!(integrator)
    end
    return SciMLBase.ContinuousCallback(condition, affect!; save_positions = (false, false))
end
