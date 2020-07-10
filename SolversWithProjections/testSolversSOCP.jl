# SOCP Solver Tests

using Plots
pyplot()

include("AL-Primal-SOCP-Solver.jl")
include("runSOCP-Setup.jl")

yStates, res = ALPrimalNewtonSOCPmain(y0, alcone, currSolveParams, false)

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
alclean = augLagQP_2Cone(thisQP, thisConstr, 1, # currSolveParams.penaltyMax,
                                zeros(lambdaSize))

yEnd = yStates[end]

sTest = [s0, yEnd.s] # [[0;0;0], s0, [20; 20; 20]]
tTest = [t0, yEnd.t] # [0, t0, 10]

xRange = -15:1:10
yRange = -6:1:10

for (si, sIter) in enumerate(sTest)
    for (ti, tIter) in enumerate(tTest)
        merit(x, y) = evalAL(alclean, SOCP_primals([x; y], sIter, tIter))

        contoursMerit = contour(xRange, yRange, merit)
        title!("Contour Plots of the Merit Function\n" *
                "With s = $sIter and t = $tIter")
        xlabel!("X")
        ylabel!("Y")
        display(contoursMerit)
        savefig(contoursMerit, "contoursMerit-$si-$ti")
    end
end


#=
Get and plot path
=#

xVecs = getXVals(yStates)
xVals = [x[1] for x in xVecs]
yVals = [x[2] for x in xVecs]

xRange = -5:0.5:0
yRange = -1:0.5:4

merit(x, y) = evalAL(alclean, SOCP_primals([x; y], yEnd.s, yEnd.t))

pltPath = plot(xVals, yVals, markershape = :circle, label = "Solver Path")
scatter!([-47/23], [38/23], markershape = :xcross, markercolor = :red,
                    label = "Minimum of Objective")
contour!(xRange, yRange, merit)
title!("Solver Path")
xlabel!("X")
ylabel!("Y")
display(pltPath)

xRange = -5:0.1:0
yRange = -1:0.1:4

pltCone = contour(xRange, yRange,
            (x, y) -> max(coneValOriginal(alcone.constraints, [x; y]), 0),
            levels = 50)
plot!(xVals, yVals, markershape = :circle, color = :blue, label = "Solver Path")
scatter!([-47/23], [38/23], markershape = :xcross, markercolor = :red,
                    label = "Minimum of Objective")
display(pltCone)

println("Tests Complete")
