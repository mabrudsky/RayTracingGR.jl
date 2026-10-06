module RayTracingGR
    import Parameters: @with_kw
    import ForwardDiff: Dual, Partials, partials
    using StaticArrays
    using KernelAbstractions
    import DifferentialEquations as DE
    
    include("RayTracing_types.jl")
    include("configuration/configuration.jl")
    include("spacetime/spacetime.jl")
    include("propagate/propagate.jl")   
    
    # Export spacetime module
    export kerr
    
    # Export spacetime functions
    export metric, metric_inverse, christoffel
    
end