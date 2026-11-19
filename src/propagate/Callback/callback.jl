# ================================================================================ #
# Integration regions: each scenario (pulsar, accretion disk, ...) is a region     #
# where the ray is integrated (one file per region in Region/), and defines        #
#   region_condition(region, spacetime, position) -> changes sign on every         #
#   surface where the ray may stop (> 0 at the start of the ray for PulsarRegion). #
#   stops_at_boundary(region, spacetime, position) -> whether that crossing stops  #
#   the ray (default: true; the disk only stops on the plane inside the disk).     #
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
function region_condition end    # region_condition(region, spacetime, position) -> changes sign where the ray may stop
function boundary_callback end   # boundary_callback(region, backend) -> callback that stops the ray at the boundary
function stops_at_boundary end   # stops_at_boundary(region, spacetime, position) -> does this sign change stop the ray?

# By default every sign change of region_condition stops the ray (e.g. PulsarRegion).
# A region where some crossings do not stop the ray (e.g. the equatorial plane outside
# an accretion disk) adds its own method.
@inline stops_at_boundary(region::AbstractRegion, spacetime::AbstractSpacetime, position::SVector{4}) = true

#------------------------------- regions ----------------------------------
include("Region/PulsarRegion.jl")
include("Region/AccretionDiskRegion.jl")

#------------------------------- backends ---------------------------------
include("boundary_callback_cpu.jl")
include("boundary_callback_gpu.jl")

# boundary_callback(region, configuration): uses configuration.backend
boundary_callback(region::AbstractRegion, configuration::Configuration) = boundary_callback(region, configuration.backend)
