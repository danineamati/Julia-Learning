# Solver tests

using Plots
gr()
include("AL-Primal-Solver.jl")
include("AL-PD-Solver.jl")

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
# Refresh the parameters
alTest = augLagQP_AffineIneq(thisQP, thisConstr, 1, zeros(size(bVec)))
xResultsALP, resResultsALP = ALPrimalNewtonQPmain(x0, alTest, currSolveParams, false)

println("Solver Complete\n")

# First up is the primal Augmented Lagrangian solver
println("2) Augemented Lagrangian Primal-Dual")
# Refresh the parameters
alTest = augLagQP_AffineIneq(thisQP, thisConstr, 1, zeros(size(bVec)))
xResultsALPD, resResultsALPD = ALPDNewtonQPmain(x0, alTest, currSolveParams, false)

println("Solver Complete\n")

plt1 = plot(calcNormError(xResultsALP), yaxis = :log, markershape = :circle,
                    label = "AL Primal Error")

plot!(getNormRes(xResultsALPD), yaxis = :log, markershape = :circle,
                    label = "AL Primal Residuals", legend = :bottomleft)

scatter!(calcNormError(xResultsALPD), yaxis = :log, markershape = :utriangle,
                    label = "AL PD Error")

scatter!(getNormRes(xResultsALPD), yaxis = :log, markershape = :utriangle,
                    label = "AL PD Residuals", legend = :bottomleft)

xlabel!("Recorded Newton Step")
ylabel!("Norm of Residual or Error")
title!("Augmented Lagrangian Performance\n" *
            "For a max $(currSolveParams.maxNewtonSteps) max Newton Steps" *
            ", for $(QPName) QP")
display(plt1)
savefig(plt1, "PrimalAL-Performance$(currSolveParams.maxNewtonSteps)NewtStep" *
                    "For $(QPName) QP - OutsideFeasibleStart")


xValsALP = [x[1] for x in xResultsALP]
yValsALP = [x[2] for x in xResultsALP]

xValsALPD = [x[1] for x in xResultsALPD]
yValsALPD = [x[2] for x in xResultsALPD]

plt2 = scatter([x0[1]], [x0[2]], markershape = :rect, label = "Initial Point")
scatter!([35/6], [-1], markershape = :xcross, label = "Optimal Point")
plot!(xValsALP, yValsALP, markershape = :circle, label = "ALP")
plot!(xValsALPD, yValsALPD, markershape = :circle, label = "ALPD")

xlabel!("X")
ylabel!("Y")
title!("Augmented Lagrangian Performance\n" *
            "For a max $(currSolveParams.maxNewtonSteps) max Newton Steps" *
            ", for $(QPName) QP")
display(plt2)
savefig(plt2, "PrimalAL-Path$(currSolveParams.maxNewtonSteps)NewtStep" *
                    "For $(QPName) QP - OutsideFeasibleStart")
