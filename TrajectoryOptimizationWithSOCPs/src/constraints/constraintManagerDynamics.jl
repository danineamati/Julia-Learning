# Dynamically aware constraints manager

include("constraintManager.jl")
include("..\\objective\\LQR_objective.jl")


"""
    constraintManager_Dynamics

Contains a list of constraints and the dual variables (lambda) that correspond
with each constraint.

Unlike [`constraintManager_Base`](@ref), this constraint manager also has the
dynamics constraints embedded such that the nearest dynamically feasible
trajectory can be determined at any given time.
"""
struct constraintManager_Dynamics
    cList::Array{constraint, 1}
    lambdaList
    dynamicsConstraint::AL_AffineEquality
    lqr::LQR_QP_Referenced
end


#############################################################
###    Get the nearest dynamically feasible trajectory    ###
#############################################################

function getNearestDynamicallyFeasibleTraj(args)
    body
end
