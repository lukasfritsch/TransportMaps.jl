
julia --project=docs/ -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.update()'

julia --project=docs/ docs/literate_docs.jl

julia --project=docs/ docs/make.jl
