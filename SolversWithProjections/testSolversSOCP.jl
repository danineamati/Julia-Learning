# SOCP Solver Tests

using Plots
pyplot()

include("AL-Primal-SOCP-Solver.jl")
include("runSOCP-Setup.jl")

yStates, res = ALPrimalNewtonSOCPmain(y0, alcone, currSolveParams, false)

xVecs = getXVals(yStates)
xVals = [x[1] for x in xVecs]
yVals = [x[2] for x in xVecs]

plt = plot(xVals, yVals, markershape = :circle)
title!("Solver Path")
xlabel!("X")
ylabel!("Y")
display(plt)

cV, aV, lV = getViolation(yStates, alcone.constraints)

pltC = plot(cV, markershape = :rect, label = "Cone Violation")
plot!(norm.(aV), markershape = :circle, label = "(Ax - b - s) Violation")
plot!(lV, markershape = :star4, label = "(c'x - d - t) Violation",
    linestyle = :dash)

title!("Constraint Violation")
xlabel!("Newton Step")
ylabel!("Violation")
display(pltC)

println("Tests Complete")
