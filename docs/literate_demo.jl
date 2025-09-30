using Literate

# Define paths
literate_dir = joinpath(@__DIR__, "literate", "Examples")
demo_dir = joinpath(@__DIR__, "..", "demo")

# Ensure demo directory exists
mkpath(demo_dir)

# Get all .jl files from literate directory
literate_files = filter(f -> endswith(f, ".jl"), readdir(literate_dir))

# Process each literate file
for file in literate_files
    input_file = joinpath(literate_dir, file)

    # Generate output filename - keep the same name for demos
    output_name = file

    println("Processing: $file -> $output_name")

    # Use Literate.script to generate clean Julia code without documentation
    Literate.script(input_file, demo_dir;
                    name=splitext(output_name)[1],  # Remove .jl extension, Literate adds it back
                    execute=false,                  # Don't execute the code
                    documenter=false,               # Don't add Documenter.jl specific code
                    credit=true)                    # Add Literate.jl credit comment

end
