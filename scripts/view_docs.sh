
#! /bin/bash

julia --project=docs/ -e 'using LiveServer; LiveServer.serve(dir = "docs/build/1")'