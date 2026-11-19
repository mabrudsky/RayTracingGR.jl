module RayTracingGR
    import ForwardDiff: Dual, Partials, partials, derivative
    using StaticArrays
    import KernelAbstractions as KA
    import SciMLBase
    import OrdinaryDiffEqVerner: Vern9
    import DiffEqGPU: GPUTsit5, EnsembleGPUKernel, GPUODEAlgorithm

    include("RayTracing_types.jl")
    include("configuration/configuration.jl")
    include("spacetime/spacetime.jl")
    include("propagate/propagate.jl")   
    
    # Export spacetime module
    export Kerr
    
    # Export spacetime functions
    export metric, metric_inverse, christoffel, coordinate_radius

    # Export integration regions (callbacks)
    export PulsarRegion, boundary_callback

    # Export configuration, equation sets and propagation of the rays
    export Configuration, GeodesicEquationsVacuum, ParallelTransportEquationsVacuum
    export propagate
    
end