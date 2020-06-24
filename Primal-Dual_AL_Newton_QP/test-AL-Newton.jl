# Test the Augemented Lagrangian Monte Carlo Style

include("augmentedLagranginaMethod.jl")
include("..\\Primal-Dual_IP_Newton_QP\\QP-Setup.jl")

QMat, cVec, AMat, bVec, x0 = QPSetup(true, true)
title!("Monte Carlo runs for AL Approach")

fObj(x) = (1/2) * x'QMat*x + cVec'x
dfdx(x) = QMat * x + cVec

rhoStart = 1
lambdaStart = zeros(size(AMat, 1))

yMin = -2
yMax = 10

xMin = -10
xMax = 10
xMCMax = 5 # To avoid overlap with the legend

xRange = xMin:0.01:xMCMax
yRange = yMin:0.01:yMax

numPoints = 5

xList = rand(xRange, numPoints)
yList = rand(yRange, numPoints)
xyList = vcat(x0', hcat(xList, yList))

# Run as Monte Carlo
println("Running Monte Carlo")
for ind in 1:size(xyList, 1)
    println("Starting run $ind")
    xStart = xyList[ind, :]
    scatter!([xStart[1]], [xStart[2]], markersize = 8,
                                label = "Initial pt $ind",
                                markershape = :rect)

    xStates = ALNewtonQPmain(xStart, fObj, dfdx, QMat, cVec, AMat, bVec,
                                    rhoStart, lambdaStart)
    xVals = [x[1] for x in xStates]
    yVals = [x[2] for x in xStates]
    plt = plot!(xVals, yVals, markershape = :circle, label = "Iterations $ind")
    ylims!(yMin, yMax)
    xlims!(xMin, xMax)
    display(plt)
    savefig("success$ind-AL-Up")
end
