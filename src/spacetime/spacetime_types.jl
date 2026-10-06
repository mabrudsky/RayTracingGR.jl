# spacetime types
abstract type AbstractKerrSpacetime <: AbstractSpacetime end

# topology types
abstract type AbstractCoordinate <: AbstractSpacetime end
struct CartesianCoordinate <: AbstractCoordinate end
struct SphericalCoordinate <: AbstractCoordinate end