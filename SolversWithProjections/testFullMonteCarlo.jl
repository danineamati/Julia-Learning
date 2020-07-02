# Full integrated solver and monte carlo
# Saves a plot at each step to generate a GIF at the end.

println()
println("##############################################")
println("Beginning Solver with Monte Carlo Capabilities")
println("##############################################")
println("Loading packages")

# For timestamp in file name
import Dates

# Need the plotting library for plotting
using Plots
# For speed, use GR
# gr()
# For nice plots, use Pyplot
pyplot()

# Import the solvers
include("AL-Primal-Solver.jl")
include("AL-PD-Solver.jl")

# Import the Monte Carlo Engine
include("montecarlo.jl")


println()
println("*****************************")
println("*   Running the AL Set-UP   *")
println("*****************************")
# Instantiate the QP, Constraints, and Augemented Lagrangian
include("runQP-Setup.jl")

println()
println("Augmented Lagrangian Instance Complete")
println()

println()
println("*******************************")
println("* Running the Monte Carlo Gen *")
println("*******************************")

println("Allowing starting points outside feasible region")
minVec = [-10; -2]
maxVec = [5; 10]
numPoints = 10
ptList = montecarlo(minVec, maxVec, numPoints)

xPt = [x[1] for x in ptList]
yPt = [x[2] for x in ptList]

plt = scatter(xPt, yPt, markershape = :rect, label = "Starting Points",
                    markersize = 6)

println()
println("*****************************")
println("*    Running the Solvers    *")
println("*****************************")


alClean = augLagQP_AffineIneq(thisQP, thisConstr, alTest.rho, zeros(size(bVec)))

xOverallBinALP = []
residOverallBinALP = []

xOverallBinALPD = []
residOverallBinALPD = []

for xIter in ptList
    # First up is the primal Augmented Lagrangian solver
    println("1) Augemented Lagrangian Primal")
    # Refresh the parameters
    alTest = augLagQP_AffineIneq(thisQP, thisConstr, 1, zeros(size(bVec)))
    xResultsALP, resStateResultsALP = ALPrimalNewtonQPmain(xIter, alTest,
                                                    currSolveParams, false)
    println("Solver Complete\n")
    # Calculates the residuals based on the last penalty parameter only
    # Remove the approximate lagrange multiplier λ → 0, using alClean
    resResultsALP = calcNormGradResiduals(alClean, xResultsALP)

    push!(xOverallBinALP, xResultsALP)
    push!(residOverallBinALP, resResultsALP)

    # First up is the primal Augmented Lagrangian solver
    println("2) Augemented Lagrangian Primal-Dual")
    # Refresh the parameters
    alTest = augLagQP_AffineIneq(thisQP, thisConstr, 1, zeros(size(bVec)))
    hResultsALPD, resStateResultsALPD = ALPDNewtonQPmain(xIter, alTest,
                                                    currSolveParams, false)
    println("Solver Complete\n")
    xResultsALPD = [[h[1]; h[2]] for h in hResultsALPD]
    # Calculates the residuals based on the last penalty parameter
    # Removes the approximate lagrange multiplier λ → 0
    resResultsALPD = calcNormGradResiduals(alClean, xResultsALPD)

    push!(xOverallBinALPD, xResultsALPD)
    push!(residOverallBinALPD, resResultsALPD)
end


println()
println("*****************************")
println("*   Plotting the Results    *")
println("*****************************")

function plotPath(xArr, thismarker = :circle, thislabel = "Path",
                        linestyle = :dash)
    xVals = [x[1] for x in xArr]
    yVals = [x[2] for x in xArr]
    plot!(xVals, yVals, markershape = thismarker, markersize = 5,
            markerstrokewidth = 0, label = thislabel)
end

function plotAllPaths(alpPath, alpdPath, filename = "Path")
    numPaths = min(size(alpPath, 1), size(alpdPath, 1))

    for i in 1:numPaths
        plotPath(alpPath[i], :circle, "AL Primal $i", :dash)
        savefig(plt, fileName * "-$i-1")
        plotPath(alpdPath[i], :star4, "AL Primal-Dual $i", :dot)
        savefig(plt, fileName * "-$i-2")
    end
end

fileName = "MCpathPlot-$(QPName)-$(currTimeFormat)"
scatter!([35/6], [-1], markershape = :xcross, label = "Optimal Point",
            legend = :outertopright, markersize = 8, markercolor = :red)

xyBuff = 2
xlabel!("X")
ylabel!("Y")
xlims!(minVec[1] - xyBuff, maxVec[1] + xyBuff * 4)
ylims!(minVec[2] - xyBuff * 4, maxVec[2] + xyBuff)
title!("Monte Carlo Simulation for $(QPName) QP")

savefig(plt, fileName * "-0")

plotAllPaths(xOverallBinALP, xOverallBinALPD, fileName)

display(plt)

currTimeFormat = Dates.format(Dates.now(), "yymmdd-HHMMSS")
savefig(plt, fileName * "-Final")
