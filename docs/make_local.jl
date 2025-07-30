using Documenter
using DocumenterCitations
using Literate
using TransportMaps

# Setup bibliography
bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

# Process Literate.jl files
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const OUTPUT_DIR = joinpath(@__DIR__, "src")

println("Processing Literate.jl files...")

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

println("Building documentation...")

# Simple local documentation build
makedocs(
    sitename = "TransportMaps.jl",
    modules = [TransportMaps],
    plugins = [bib],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Examples" => [
            "Banana Distribution" => "Examples/banana_example.md"
        ],
        "API Reference" => "api.md",
        "References" => "references.md"
    ],
    checkdocs = :none,
)
