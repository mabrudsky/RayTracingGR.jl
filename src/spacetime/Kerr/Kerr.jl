module Kerr
using StaticArrays
using ..RayTracingGR: AbstractKerrSpacetime, CartesianCoordinate, SphericalCoordinate

import ..RayTracingGR: metric, metric_inverse, numtype, coordinate_radius # Bring in "metric", "metric_inverse", "numtype" and "coordinate_radius" so this module adds a new case to each instead of accidentally creating a separate, disconnected function.
#------------------------------- spacetime --------------------------------- 
include("KerrSchildCoordinates.jl")
include("BoyerLindquistCoordinates.jl")

end # module Kerr   