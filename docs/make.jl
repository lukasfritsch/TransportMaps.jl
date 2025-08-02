using Documenter
using DocumenterCitations
using Literate
using TransportMaps

# Setup bibliography
bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

# Process Literate.jl files
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const OUTPUT_DIR = joinpath(@__DIR__, "src")

# Process getting started guide
Literate.markdown(
    joinpath(LITERATE_DIR, "getting_started.jl"),
    OUTPUT_DIR;
    documenter = true,
    credit = false
)

# Process banana example
Literate.markdown(
    joinpath(LITERATE_DIR, "banana_example.jl"),
    joinpath(OUTPUT_DIR, "Examples");
    documenter = true,
    credit = false
)

# Process BOD example
Literate.markdown(
    joinpath(LITERATE_DIR, "bod_example.jl"),
    joinpath(OUTPUT_DIR, "Examples");
    documenter = true,
    credit = false
)

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
        "Getting Started" => "getting_started.md",
        "Examples" => [
            "Banana Distribution" => "Examples/banana_example.md",
            "BOD Parameter Estimation" => "Examples/bod_example.md"
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
