using RayTracingGR
using Test

@testset "RayTracing.jl" begin
    include("metric/christoffel_tests_KerrSchildCoordinates.jl")
    include("metric/kerr_schild_tests.jl")
    include("propagate/ensemble_tests.jl")
end
