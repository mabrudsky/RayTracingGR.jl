# The physical model (vacuum, plasma, ...) is defined by the equation set type
# (e.g. GeodesicEquationsVacuum). If a future equation set needs parameters, they
# live inside its struct and it is added here as a field.
@with_kw struct IntegrationParameters{S <: AbstractSpacetime} <: AbstractConfiguration
    spacetime::S
end

@with_kw struct Configuration{N, T <: Union{Float32,Float64}, S <: AbstractSpacetime, E <: AbstractEquationSet, B <: KernelAbstractions.Backend} <: AbstractConfiguration
    spacetime::S
    equations::E
    backend::B
    u0::SVector{N,T}
    tspan::Tuple{T,T}
end
