"""
    SolversWIthProjections

A mini-package that solves simple SOCP problems.

"""
module SolversWithProjections

using LinearAlgebra, Plots

# structs
export solverParams, objectiveQP, SOCP_Primals, augLagQP_2Cone

# functions
export ALPrimalNewtonSOCPmain

include("AL-Primal-SOCP-Solver.jl")
include("SOCP-Setup-Simple.jl")
include("trustRegion.jl")

end
