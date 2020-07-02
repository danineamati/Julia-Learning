# Solver tests

using Plots
gr()
include("AL-Primal-Solver.jl")
include("runQP-Setup.jl")

println()
println("*****************************")
println("* Running the Solver Tests  *")
println("*****************************")

function getNormRes(resArr, floor = 10^(-20))
    # Take the norm to get non-negative.
    # Take max with floor to get positive.
    return max.(norm.(resArr), floor)
end

function calcNormError(xArr)
    xs = [x[1] for x in xArr]
    ys = [x[2] for x in xArr]

    # Notice "Magic Numbers" of the correct value to prevent storing it as a
    # variable (for now. i.e. will be changed)
    # [-5/4; 5] - UP configuration
    # [-47/23; 38/23] - Interior configuration
    # [35/6; -1] - Far Exterior configuration
    res = [norm([xs[ind]; ys[ind]] - [35/6; -1]) for ind in 1:size(xs,1)]
    return res
end

# First up is the primal Augmented Lagrangian solver
println("1) Augemented Lagrangian Primal")
xResults, resResults = ALPrimalNewtonQPmain(x0, alTest, currSolveParams, false)

println("Solver Complete")

plt1 = plot(calcNormError(xResults), yaxis = :log, markershape = :circle,
                    label = "AL Primal Error")

plot!(getNormRes(resResults), yaxis = :log, markershape = :circle,
                    label = "AL Primal Residuals", legend = :bottomleft)

xlabel!("Recorded Newton Step")
ylabel!("Norm of Residual or Error")
title!("Primal Augmented Lagrangian Performance\n" *
            "For a max $(currSolveParams.maxNewtonSteps) max Newton Steps" *
            ", for $(QPName) QP")
display(plt1)
savefig(plt1, "PrimalAL-Performance$(currSolveParams.maxNewtonSteps)NewtStep" *
                    "For $(QPName) QP - OutsideFeasibleStart")
