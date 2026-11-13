module kerr
using StaticArrays
using ..RayTracingGR: AbstractSpacetime, AbstractKerrSpacetime, AbstractCoordinate, CartesianCoordinate, SphericalCoordinate
using ..RayTracingGR: AbstractChristoffel, ChristoffelTag, _Γcomp, _christoffel_ForwardDiff

import ..RayTracingGR: metric, metric_inverse, christoffel, numtype # Bring in "metric", "metric_inverse" and "christoffel" so this module adds a new case to each instead of accidentally creating a separate, disconnected function.
#------------------------------- spacetime --------------------------------- 
include("KerrSchildCoordinates.jl")

export metric, metric_inverse, christoffel

end # module kerr   