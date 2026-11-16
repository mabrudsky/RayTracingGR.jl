# Region where a ray is integrated (pulsar, accretion disk, ...). Each scenario is a concrete
# isbits struct in Callback/callback.jl with its own region_condition.
abstract type AbstractRegion <: AbstractPropagate end
