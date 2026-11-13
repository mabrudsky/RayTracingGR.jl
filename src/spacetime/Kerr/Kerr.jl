module Kerr
using StaticArrays
using ..RayTracingGR: AbstractKerrSpacetime, CartesianCoordinate, _christoffel_ForwardDiff

import ..RayTracingGR: metric, metric_inverse, christoffel, numtype # Bring in "metric", "metric_inverse" and "christoffel" so this module adds a new case to each instead of accidentally creating a separate, disconnected function.
#------------------------------- spacetime --------------------------------- 
include("KerrSchildCoordinates.jl")

end # module Kerr   