# ================================================================================ #
# Integration regions: each scenario (pulsar, accretion disk, ...) is a region     #
# where the ray is integrated (one file per region in Region/), and defines        #
#   region_condition(region, spacetime, position) -> > 0 while the ray is inside   #
#   the region, and changes sign at its boundary.                                  #
# boundary_callback(region, backend) turns it into a callback that finds the       #
# crossing and stops the ray there (terminate!). The type of callback is chosen    #
# from the backend (one file per backend):                                         #
#   CPU() -> ContinuousCallback                       (boundary_callback_cpu.jl)   #
#   GPU   -> DiscreteCallback + bisection             (boundary_callback_gpu.jl)   #
# ================================================================================ #

#= Declare "region_condition" and "boundary_callback" here first, with no
   implementation yet, so that every region and every backend adds to these
   SAME functions instead of each one creating its own.
=#
function region_condition end    # region_condition(region, spacetime, position) -> > 0 inside, 0 at the boundary, < 0 outside
function boundary_callback end   # boundary_callback(region, backend) -> callback that stops the ray at the boundary

#------------------------------- regions ----------------------------------
include("Region/PulsarRegion.jl")

#------------------------------- backends ---------------------------------
include("boundary_callback_cpu.jl")
include("boundary_callback_gpu.jl")

# boundary_callback(region, configuration): uses configuration.backend
boundary_callback(region::AbstractRegion, configuration::Configuration) = boundary_callback(region, configuration.backend)
