using Documenter

# push!(LOAD_PATH,"../../")

# include("C:\\Users\\Daniel N\\Desktop\\GitHub\\Julia-Learning\\SolversWithProjections\\SolversWithProjections")

include("..\\src\\SOCP_Rocket_TrajOpt.jl")


# using SolversWithProjections

makedocs(modules = [SOCP_Rocket_TrajOpt],
         sitename = "SOCP Trajectory Optimization Documentation",
         pages = ["Home" => "index.md",
                  "User Interface" => "ui.md"])
