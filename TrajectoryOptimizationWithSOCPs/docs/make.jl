using Documenter

# push!(LOAD_PATH,"../../")

# include("C:\\Users\\Daniel N\\Desktop\\GitHub\\Julia-Learning\\SolversWithProjections\\SolversWithProjections")

include("..\\src\\SOCP_Rocket_TrajOpt.jl")


# using SolversWithProjections

makedocs(modules = [SOCP_Rocket_TrajOpt],
         sitename = "SOCP Trajectory Optimization Documentation",
         pages = [
                  "Home" => "index.md",
                  "Using the Module" => Any[
                        "Overview" => "UI/ui.md",
                        "(1) Rockets" => "UI/rocket_ui.md",
                        "(2) Objective" => "UI/objective_ui.md",
                        "(3) Constraints" => "UI/constraints_ui.md",
                        "(4) Augmented Lagrangian" => "UI/augLag_ui.md"
                  ]
                 ],
         format = Documenter.HTML(prettyurls = false))
