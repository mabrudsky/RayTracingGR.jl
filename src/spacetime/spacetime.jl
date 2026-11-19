include("spacetime_types.jl")
include("christoffel.jl")

#= Declare "metric" and "metric_inverse" here first, with no
   implementation yet, so that every spacetime module (Kerr,
   Schwarzschild, etc.) adds to these SAME functions instead of
   each one creating its own.
=#
function metric end
function metric_inverse end
function numtype end        # numtype(spacetime) -> Float32 or Float64 (precision of the spacetime parameters)
function coordinate_radius end   # coordinate_radius(spacetime, point) -> radius associated with each coordinate system (used for the cuts)
function coordinate_z end        # coordinate_z(spacetime, point) -> height above the equatorial plane, z = r cos θ (used for the disk)

include("Kerr/Kerr.jl")