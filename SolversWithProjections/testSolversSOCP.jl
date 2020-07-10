# SOCP Solver Tests

using Plots
pyplot()

include("AL-Primal-SOCP-Solver.jl")
include("runSOCP-Setup.jl")

yStates, res = ALPrimalNewtonSOCPmain(y0, alcone, currSolveParams, false)

#=
Get and plot path
=#

xVecs = getXVals(yStates)
xVals = [x[1] for x in xVecs]
yVals = [x[2] for x in xVecs]

plt = plot(xVals, yVals, markershape = :circle)
title!("Solver Path")
xlabel!("X")
ylabel!("Y")
display(plt)

#=
Calculate and plot the constraint violation
=#

cV, aV, lV = getViolation(yStates, alcone.constraints)

pltC = plot(cV, markershape = :rect, label = "Cone Violation")
plot!(norm.(aV), markershape = :circle, label = "(Ax - b - s) Violation")
plot!(lV, markershape = :star4, label = "(c'x - d - t) Violation",
    linestyle = :dash)

title!("Constraint Violation")
xlabel!("Newton Step")
ylabel!("Violation")
display(pltC)

#=
Calculate and plot the residuals
=#
alclean = augLagQP_2Cone(thisQP, thisConstr, currSolveParams.penaltyMax,
                                zeros(lambdaSize))

sTest = [[0;0;0], s0, [20; 20; 20]]
tTest = [0, t0, 10]

for (si, sIter) in enumerate(sTest)
    for (ti, tIter) in enumerate(tTest)
        merit(x, y) = evalAL(alclean, SOCP_primals([x; y], sIter, tIter))

        xRange = -15:1:10
        yRange = -6:1:10

        contoursMerit = contour(xRange, yRange, merit)
        title!("Contour Plots of the Merit Function\n" *
                "With s = $sIter and t = $tIter")
        xlabel!("X")
        ylabel!("Y")
        display(contoursMerit)
        savefig(contoursMerit, "contoursMerit-$si-$ti")
    end
end

println("Tests Complete")
