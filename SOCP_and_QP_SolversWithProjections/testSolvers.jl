# Solver tests

using Plots
pyplot()
include("AL-Primal-Solver.jl")
include("AL-PD-Solver.jl")

include("runQP-Setup.jl")

println()
println("*****************************")
println("* Running the Solver Tests  *")
println("*****************************")

function calcNormError(xArr)
    xs = [x[1] for x in xArr]
    ys = [x[2] for x in xArr]

    # Notice "Magic Numbers" of the correct value to prevent storing it as a
    # variable (for now. i.e. will be changed)
    # [-5/4; 5] - UP configuration
    # [-47/23; 38/23] - Interior configuration
    # [35/6; -1] - Far Exterior configuration
    res = [norm([xs[ind]; ys[ind]] - [-5/4; 5]) for ind in 1:size(xs,1)]
    return res
end

# First up is the primal Augmented Lagrangian solver
println("1) Augemented Lagrangian Primal")
# Refresh the parameters
alTest = augLagQP_AffineIneq(thisQP, thisConstr, 1, zeros(size(bVec)))
xResultsALP, resStateResultsALP = ALPrimalNewtonQPmain(x0, alTest,
                                                currSolveParams, false)
# Calculates the residuals based on the last penalty parameter
# Removes the approximate lagrange multiplier λ → 0
alClean = augLagQP_AffineIneq(thisQP, thisConstr, alTest.rho, zeros(size(bVec)))
resResultsALP = calcNormGradResiduals(alClean, xResultsALP)

println("Solver Complete\n")

# First up is the primal Augmented Lagrangian solver
println("2) Augemented Lagrangian Primal-Dual")
# Refresh the parameters
alTest = augLagQP_AffineIneq(thisQP, thisConstr, 1, zeros(size(bVec)))
hResultsALPD, resStateResultsALPD = ALPDNewtonQPmain(x0, alTest,
                                                currSolveParams, false)
xResultsALPD = [[h[1]; h[2]] for h in hResultsALPD]
# Calculates the residuals based on the last penalty parameter
# Removes the approximate lagrange multiplier λ → 0
alClean = augLagQP_AffineIneq(thisQP, thisConstr, alTest.rho, zeros(size(bVec)))
resResultsALPD = calcNormGradResiduals(alClean, xResultsALPD)

println("Solver Complete\n")

plt1 = plot(calcNormError(xResultsALP), yaxis = :log, markershape = :circle,
                    label = "AL Primal Error")

scatter!(calcNormError(xResultsALPD), yaxis = :log, markershape = :utriangle,
                    label = "AL PD Error")

xlabel!("Recorded Newton Step")
ylabel!("Norm of Error")
title!("Augmented Lagrangian Performance\n" *
            "For a max $(currSolveParams.maxNewtonSteps) max Newton Steps" *
            ", for $(QPName) QP")

display(plt1)
savefig(plt1, "PrimalAL-Performance$(currSolveParams.maxNewtonSteps)NewtStep" *
                    "For $(QPName) QP-Error")

plt2 = plot(resResultsALP, yaxis = :log, markershape = :circle,
                    label = "AL Primal Residuals", legend = :bottomleft)

scatter!(resResultsALPD, yaxis = :log, markershape = :utriangle,
                    label = "AL PD Residuals", legend = :bottomleft)

xlabel!("Recorded Newton Step")
ylabel!("Norm of Residual")
title!("Augmented Lagrangian Performance\n" *
            "For a max $(currSolveParams.maxNewtonSteps) max Newton Steps" *
            ", for $(QPName) QP")
display(plt2)
savefig(plt2, "PrimalAL-Residual$(currSolveParams.maxNewtonSteps)NewtStep" *
                    "For $(QPName) QP")


xValsALP = [x[1] for x in xResultsALP]
yValsALP = [x[2] for x in xResultsALP]

xValsALPD = [x[1] for x in xResultsALPD]
yValsALPD = [x[2] for x in xResultsALPD]

plt3 = scatter([x0[1]], [x0[2]], markershape = :rect, label = "Initial Point",
                markersize = 8)
scatter!([-47/23], [38/23], markershape = :xcross, label = "Optimal Point",
                markersize = 8)
plot!(xValsALP, yValsALP, markershape = :circle, label = "ALP")
plot!(xValsALPD, yValsALPD, markershape = :circle, label = "ALPD")

xlabel!("X")
ylabel!("Y")
title!("Augmented Lagrangian Performance\n" *
            "For a max $(currSolveParams.maxNewtonSteps) max Newton Steps" *
            ", for $(QPName) QP")
display(plt3)
savefig(plt3, "PrimalAL-Path$(currSolveParams.maxNewtonSteps)NewtStep" *
                    "For $(QPName) QP")

consVioALP = [getNormToProjVals(alClean.constraints, x) for x in xResultsALP]
consVioALPD = [getNormToProjVals(alClean.constraints, x) for x in xResultsALPD]

function safeNorm(arr, vMin = 10^-20, p = 2)
    return max.(norm.(arr, p), vMin)
end

plt4 = plot(safeNorm(consVioALP), yaxis = :log, markershape = :circle,
                    label = "ALP")
plot!(safeNorm(consVioALPD), yaxis = :log, markershape = :utriangle,
                    label = "ALPD")

xlabel!("Recorded Newton Step")
ylabel!("Norm of Residual")
title!("Augmented Lagrangian Constriant Violation\n" *
            "For a max $(currSolveParams.maxNewtonSteps) max Newton Steps" *
            ", for $(QPName) QP")
display(plt4)
savefig(plt4, "PrimalAL-ConstraintViolation" *
                    "$(currSolveParams.maxNewtonSteps)NewtStep" *
                    "For $(QPName) QP")
