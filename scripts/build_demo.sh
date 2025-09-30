
#! /bin/bash

julia --project=docs/ -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.update()'

julia --project=docs/ docs/literate_demo.jl