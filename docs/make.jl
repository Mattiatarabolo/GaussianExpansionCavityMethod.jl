using GaussianExpansionCavityMethod
using Documenter

DocMeta.setdocmeta!(GaussianExpansionCavityMethod, :DocTestSetup, :(using GaussianExpansionCavityMethod); recursive=true)

makedocs(;
    modules=[GaussianExpansionCavityMethod],
    authors="Mattia Tarabolo <mattia.tarabolo@gmail.com> and contributors",
    sitename="GaussianExpansionCavityMethod.jl",
    format=Documenter.HTML(;
        canonical="https://Mattiatarabolo.github.io/GaussianExpansionCavityMethod.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Mattiatarabolo/GaussianExpansionCavityMethod.jl",
    devbranch="main",
)
