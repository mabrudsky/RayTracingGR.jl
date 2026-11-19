using RayTracingGR
using Test

@testset "RayTracing.jl" begin
    include("metric/christoffel_tests_KerrSchildCoordinates.jl")
    include("metric/kerr_schild_tests.jl")
    include("metric/christoffel_tests_BoyerLindquistCoordinates.jl")
    include("metric/boyer_lindquist_tests.jl")
    include("propagate/ensemble_tests.jl")
    include("propagate/callback_tests.jl")
    include("propagate/solve_tests.jl")
    include("propagate/constants_of_motion_tests.jl")
end
