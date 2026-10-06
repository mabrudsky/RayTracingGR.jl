abstract type AbstractRayTracing end

abstract type AbstractPropagate <: AbstractRayTracing end
abstract type AbstractSpacetime <: AbstractRayTracing end
abstract type AbstractConfiguration <: AbstractRayTracing end
abstract type AbstractPhysicalModel <: AbstractRayTracing end

abstract type AbstractEquationSet <: AbstractPropagate end