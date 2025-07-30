#!/usr/bin/env julia

# Simple script to build documentation locally
# Usage: julia docs/build_docs.jl

println("Building TransportMaps.jl documentation...")
println("=" ^ 50)

cd(joinpath(@__DIR__))

# Activate the docs environment
using Pkg
Pkg.activate(".")

# Build the documentation
include("make_local.jl")

println("=" ^ 50)
println("Documentation built successfully!")
println("Open docs/build/index.html in your browser to view the results.")
