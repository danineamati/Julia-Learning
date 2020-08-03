# Plot the total constraint violation of the course of the solver

include("..\\constraints\\constraintManager.jl")

function plotConstraintViolation(cM::constraintManager, trajList,
                                 penalty::Float64 = 1.0)
    cVList = getConstraintViolationList(cM, trajList, penalty)

    pltCV = plot(cVList, markershape = :square)
    title!("Constraint Violation")
    ylabel!("Constraint Violation at Penalty of $penalty")
    xlabel!("Step")

    if typeof(cM) == constraintManager_Dynamics
        cVListDyn = [safeNorm(norm(evalAffineEq(cM, traj))) for traj in trajList]
        pltCV2 = plot(cVListDyn, markershape = :square, markercolor = :darkred,
                    linecolor = :darkred, yaxis = :log)
        title!("Dynamics Constraint Violation")
        ylabel!("Constraint Violation at Penalty of $penalty")
        xlabel!("Step")
        display(pltCV2)
    end

    return pltCV
end
