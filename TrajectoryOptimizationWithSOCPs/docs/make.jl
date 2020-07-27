using Documenter

# push!(LOAD_PATH,"../../")

# include("C:\\Users\\Daniel N\\Desktop\\GitHub\\Julia-Learning\\SolversWithProjections\\SolversWithProjections")

include("..\\src\\SOCP_TrajOpt.jl")


# using SolversWithProjections

makedocs(modules = [SOCP_TrajOpt],
         sitename = "SOCP Trajectory Optimization Documentation")
