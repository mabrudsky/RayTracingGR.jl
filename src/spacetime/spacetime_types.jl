# spacetime types
abstract type AbstractKerrSpacetime <: AbstractSpacetime end

# topology types
abstract type AbstractCoordinate <: AbstractRayTracing end   # a coordinate system is not a spacetime
struct CartesianCoordinate <: AbstractCoordinate end
struct SphericalCoordinate <: AbstractCoordinate end