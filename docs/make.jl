using Documenter
using DocumenterCitations
using Literate
using TransportMaps

# Setup bibliography
bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

# Process Literate.jl files
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const OUTPUT_DIR = joinpath(@__DIR__, "src")

# Automatically process all literate files in subdirectories of LITERATE_DIR
if isdir(LITERATE_DIR)
    for subdir in sort(filter(name -> isdir(joinpath(LITERATE_DIR, name)) && !startswith(name, "."), readdir(LITERATE_DIR)))
        srcdir = joinpath(LITERATE_DIR, subdir)
        dest = joinpath(OUTPUT_DIR, subdir)
        mkpath(dest)

        files = sort(filter(name ->
                !startswith(name, ".") && endswith(name, ".jl"),
            readdir(srcdir)))

        for fname in files
            src = joinpath(srcdir, fname)
            Literate.markdown(src, dest; documenter = true, credit = true)
        end
    end
end

makedocs(
    sitename = "TransportMaps.jl",
    authors="Lukas Fritsch and Jan Grashorn",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://lukasfritsch.github.io/TransportMaps.jl",
        assets = String[],
        repolink = "https://github.com/lukasfritsch/TransportMaps.jl",
    ),
    modules = [TransportMaps],
    plugins = [bib],
    pages = [
        "Home" => "index.md",
        "Manuals" => [
            "Getting Started" => "Manuals/getting_started.md",
            "Basis Functions" => "Manuals/basis_functions.md",
            "Map Parameterization" => "Manuals/map_parameterization.md",
            "Quadrature Methods" => "Manuals/quadrature_methods.md",
            "Optimization" => "Manuals/optimization.md",
            "Conditional Densities and Samples" => "Manuals/conditional_densities.md",
        ],
        "Examples" => [
            "Banana: Map from Density" => "Examples/banana_mapfromdensity.md",
            "Banana: Map from Samples" => "Examples/banana_mapfromsamples.md",
            "Bayesian Inference: BOD" => "Examples/bod_bayesianinference.md",
            "Adaptive Transport Map" => "Examples/banana_adaptive.md",
        ],
        "API" => "api.md",
        "References" => "references.md"
    ],
    repo = "https://github.com/lukasfritsch/TransportMaps.jl/blob/{commit}{path}#{line}",
    checkdocs = :none,  # Changed from :exports to :none since we don't have docstrings yet
    doctestfilters = [r"Ptr{0x[0-9a-f]+}"],
)

deploydocs(
    repo = "github.com/lukasfritsch/TransportMaps.jl.git",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
)
