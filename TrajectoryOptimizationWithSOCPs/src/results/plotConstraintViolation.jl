# Plot the total constraint violation of the course of the solver

include("..\\constraints\\constraintManager.jl")

function plotConstraintViolation(cM::constraintManager, trajList,
                                 penalty::Float64 = 1.0)
    cVList = getConstraintViolationList(cM, trajList, penalty)

    pltCV = plot(cVList, markershape = :square)
    title!("Constraint Violation per Newton Step")
    ylabel!("Constraint Violation at Penalty of $penalty")
    xlabel!("Newton Step")

    return pltCV
end
