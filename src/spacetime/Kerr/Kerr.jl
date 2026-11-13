module Kerr
using StaticArrays
using ..RayTracingGR: AbstractKerrSpacetime, CartesianCoordinate

import ..RayTracingGR: metric, metric_inverse, numtype # Bring in "metric", "metric_inverse" and "numtype" so this module adds a new case to each instead of accidentally creating a separate, disconnected function.
#------------------------------- spacetime --------------------------------- 
include("KerrSchildCoordinates.jl")

end # module Kerr   