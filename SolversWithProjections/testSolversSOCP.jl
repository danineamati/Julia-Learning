# SOCP Solver Tests

using Plots
pyplot()

include("AL-Primal-SOCP-Solver.jl")
include("runSOCP-Setup.jl")

println("\n###################")
println("# Beginning Solve #")
println("###################")
yStates, res = ALPrimalNewtonSOCPmain(y0, alcone, currSolveParams, true)
yEnd = yStates[end]

println("***************")
print("First state: ")
println(SOCP_primals(x0, s0, t0))
print("Last state: ")
println(yEnd)

println("\n###################")
println("# Completed Solve #")
println("###################")

# Choose which to plot:
plotConVio = true
plotConVioLog = true
plotContoursST = false
plotResid = false
plotPathMerit = false
plotPathConstraints = true


function safeNorm(arr, vMin = 10^-20, p = 2)
    return max.(norm.(arr, p), vMin)
end

#=
Calculate and plot the constraint violation
=#

if plotConVio

    cV, aV, lV = getViolation(yStates, alcone.constraints)

    pltC = plot(cV, markershape = :rect, label = "Cone Violation")
    for ai in 1:size(s0, 1)
        aVi = [aVec[ai] for aVec in aV]
        plot!(aVi, markershape = :circle,
                    label = "(Ax - b - s) Violation $ai")
    end
    plot!(lV, markershape = :star4, label = "(c'x - d - t) Violation",
        linestyle = :dash)

    title!("Constraint Violation - Start at $x0")
    xlabel!("Newton Step")
    ylabel!("Violation")
    display(pltC)
    savefig(pltC, "ConstraintViolation")
end

if plotConVioLog
    # Repeat of the above but with log10

    pltClog10 = plot(safeNorm(cV), markershape = :rect, label = "Cone Violation",
                            yaxis = :log)
    for ai in 1:size(s0, 1)
        aVi = [aVec[ai] for aVec in aV]
        plot!(safeNorm(aVi), markershape = :circle,
                    label = "||Ax - b - s|| Violation $ai")
    end
    plot!(safeNorm(lV), markershape = :star4,
                    label = "||c'x - d - t|| Violation",
                    linestyle = :dash, yaxis = :log)

    title!("Norm of Constraint Violation (Log 10 Scale)\nStart at $x0")
    xlabel!("Newton Step")
    ylabel!("Violation")
    display(pltClog10)
    savefig(pltClog10, "ConstraintViolationLog10")
end

#=
Calculate and plot the residuals
=#
cleanPenalty = currSolveParams.penaltyMax
alclean = augLagQP_2Cone(thisQP, thisConstr, cleanPenalty, zeros(lambdaSize))


if plotContoursST

    sTest = [s0, yEnd.s] # [[0;0;0], s0, [20; 20; 20]]
    tTest = [t0, yEnd.t] # [0, t0, 10]

    xRange = -15:1:10
    yRange = -6:1:10

    for (ind, (sIter, tIter)) in enumerate(zip(sTest, tTest))
        merit(x, y) = evalAL(alclean, SOCP_primals([x; y], sIter, tIter))

        contoursMerit = contour(xRange, yRange, merit)
        title!("Contour Plots of the Merit Function\n" *
                "With s = $sIter and t = $tIter")
        xlabel!("X")
        ylabel!("Y")
        display(contoursMerit)
        savefig(contoursMerit, "contoursMerit-$ind")
    end
end

if plotResid
    pltRes = plot(calcNormGradResiduals(alclean, yStates), markershape = :circle,
                        yaxis = :log)
    title!("Norm of the Residuals for penalty of ρ = $cleanPenalty")
    xlabel!("Newton Step")
    ylabel!("Norm of the Residual")
    display(pltRes)
    savefig(pltRes, "Residuals")
end

#=
Get and plot path
=#

xVecs = getXVals(yStates)
xVals = [x[1] for x in xVecs]
yVals = [x[2] for x in xVecs]


if plotPathMerit
    merit(x, y) = evalAL(alclean, SOCP_primals([x; y], yEnd.s, yEnd.t))
    xRange = -5:0.5:0
    yRange = -1:0.5:4
    pltPath = plot(xVals, yVals, markershape = :circle, label = "Solver Path")
    scatter!([-47/23], [38/23], markershape = :xcross, markercolor = :red,
                        label = "Minimum of Objective")
    contour!(xRange, yRange, merit, label = "Merit Function")
    title!("Solver Path")
    xlabel!("X")
    ylabel!("Y")
    display(pltPath)
    savefig(pltPath, "meritFuncEnd")
end

if plotPathConstraints
    xRange = -5:0.1:0
    yRange = -1:0.1:4

    pltCone = contour(xRange, yRange,
                (x, y) -> fObjQP(alcone.obj, [x; y]), label = "Objective")
    contour!(xRange, yRange,
                (x, y) -> max(coneValOriginal(alcone.constraints, [x; y]), 0),
                levels = 50, label = "Feasible Region")
    # scatter!([-47/23], [38/23], markershape = :xcross,
    #                     markercolor = :red, markersize = 8,
    #                     label = "Minimum of Objective")
    scatter!([x0[1]], [x0[2]], markershape = :rect,
                        markercolor = :yellow, markersize = 8,
                        label = "Starting Point")
    plot!(xVals, yVals, markershape = :circle, color = "#1f77b4",
                label = "Solver Path")
    title!("Solver Path")
    xlabel!("X")
    ylabel!("Y")
    display(pltCone)
    savefig(pltCone, "FeasibleRegionAndPath")
end

println("Tests Complete")
