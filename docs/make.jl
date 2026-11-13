using RayTracingGR
using Documenter

DocMeta.setdocmeta!(RayTracingGR, :DocTestSetup, :(using RayTracingGR); recursive=true)

makedocs(;
    modules=[RayTracingGR],
    authors="Abrudsky Martín Ezequiel <mabrudsky07@email.com>",
    sitename="RayTracingGR.jl",
    format=Documenter.HTML(;
        canonical="https://mabrudsky.github.io/RayTracingGR.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mabrudsky/RayTracingGR.jl",
    devbranch="main",
)
