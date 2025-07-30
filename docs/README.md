# Documentation for TransportMaps.jl

This directory contains the documentation setup for TransportMaps.jl using Documenter.jl and Literate.jl.

## Structure

```
docs/
├── build_docs.jl          # Convenient script to build docs locally
├── make.jl                # Full documentation build (with deployment)
├── make_local.jl          # Local documentation build (no deployment)
├── Project.toml           # Dependencies for documentation
├── literate/              # Literate.jl source files
│   ├── getting_started.jl # Getting started guide (executable)
│   └── banana_example.jl  # Banana distribution example (executable)
└── src/                   # Generated and manual documentation files
    ├── index.md           # Main landing page
    ├── api.md             # API reference
    ├── getting_started.md # Generated from literate/getting_started.jl
    └── Examples/
        └── banana_example.md # Generated from literate/banana_example.jl
```

## Building Documentation

### Quick Local Build

The easiest way to build the documentation locally:

```bash
julia docs/build_docs.jl
```

### Manual Build

For more control:

```bash
cd docs
julia --project=. make_local.jl
```

### Full Build (with deployment)

```bash
cd docs
julia --project=. make.jl
```

## Working with Literate.jl

The getting started guide and examples are created using Literate.jl, which allows us to write executable Julia code with embedded documentation.

### File Format

Literate.jl files use special comment syntax:
- `# ` - Regular text/markdown
- `#-` - Section break
- `#=` ... `=#` - Multi-line comments
- `##` - Hide from output
- No prefix - Julia code (executed and shown)

### Regenerating Markdown

The `.md` files in `src/` are automatically generated from the `.jl` files in `literate/` during the build process. Don't edit the generated markdown files directly!

## Examples

To add a new example:

1. Create a new `.jl` file in `literate/`
2. Use Literate.jl comment syntax
3. Add the processing step to `make.jl` and `make_local.jl`
4. Add to the pages list in the `makedocs()` call

## Dependencies

The documentation requires:
- Documenter.jl (documentation generation)
- Literate.jl (executable documentation)
- TransportMaps.jl (the package itself)
- Additional packages used in examples (Distributions, Plots, etc.)

## Deployment

The documentation is set up for automatic deployment to GitHub Pages via the `deploydocs()` function in `make.jl`. This should be run in CI or manually for releases.