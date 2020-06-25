# Test the Newton Update part of the Primal-Dual Scheme

include("primal-dual-AL-Newton-QP.jl")


println()
println("Testing Newton Step on test matrix")
matA = [4 5 9; 3 2 1; 0 9 10]
matB = [2 4 5; 8 6 7; 1 4 2]
matC = [8 5 7; 3 1 6; 4 9 2]
matD = [4 5 2; 8 6 4; 1 2 3]

rVec = [3; 4; 5; 9; 8; 7]

nextStep = newtonStep(matA, matB, matC, matD, rVec)
# display(nextStep)
# println("Expected: ")
expectedStep = inv([matA matB; matC matD]) * rVec
# display(expectedStep)
println("Error: ")
display(nextStep - expectedStep)

println()
println("Setting Up the QP")
QMat, cVec, AMat, bVec, x0 = QPSetup(true, true)

fObj(x) = (1/2) * x'QMat*x + cVec'x
dfdx(x) = QMat * x + cVec

rho = 1
nu = zeros(size(AMat, 1))
lambda = ones(size(AMat, 1))

# φ(x) = f(x) + (ρ/2) c(x)'c(x) + λ c(x)
phiFun(x) = fObj(x) + (rho / 2) * cPlus(AMat, x, bVec)'cPlus(AMat, x, bVec) +
                lambda' * cPlus(AMat, x, bVec)
dPhidx(x) = getQPgradPhiAL(x, QMat, cVec, AMat, bVec, rho, lambda)

println("Objective: f(x) = (1/2) x'Qx + c'x")
println("Q = $QMat")
println("c = $cVec")
println("Constraints: Ax ≤ b")
println("A = $AMat")
println("b = $bVec")
println("Initial Starting Point: $x0")

if false
    println()
    println("Testing r Vec from QP-Setup")

    # getQPrVecAL(Q, c, A, b, x, lambda, rho, nu)
    rVec = getQPrVecAL(QMat, cVec, AMat, bVec, x0, lambda, rho, nu)

    # getQPGradrVecAL(Q, A, b, x, rho)
    grA, grB, grC, grD = getQPGradrVecAL(QMat, AMat, bVec, x0, rho)
    println("r vec = $rVec")
    println("∇r =")
    display([grA grB; grC grD])

    dirNewton = -newtonStep(grA, grB, grC, grD, rVec)
    println()
    println("First Newton Step: ")
    display(dirNewton)

    # backtrackLineSearch(xInit, dirΔ, f, dfdx, paramA, paramB, verbose = false)
    paramA = 0.1
    paramB = 0.5
    x0LS, stepLS = backtrackLineSearch(x0, dirNewton[1:2], phiFun, dPhidx,
                                    paramA, paramB)
    xDed = stepLS * dirNewton[1:2]
    println("Final Deduced Step Direction: $xDed at α = $stepLS")
end

# Now running the full function
if false
    println()
    println("Full Augmented Lagrangian Method")

    xStates = pdALNewtonQPmain(QMat, cVec, AMat, bVec, x0, lambda, rho, nu,
                    fObj, dfdx, 0.1, 0.5, true)

    display(xStates)
end

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

rho = 1
nu = zeros(size(AMat, 1))
lambda = ones(size(AMat, 1))

# Run as Monte Carlo
println("Running Monte Carlo")
for ind in 1:size(xyList, 1)
    println("Starting run $ind")
    xStart = xyList[ind, :]
    scatter!([xStart[1]], [xStart[2]], markersize = 8,
                                label = "Initial pt $ind",
                                markershape = :rect)

    xStates = pdALNewtonQPmain(QMat, cVec, AMat, bVec, xStart, lambda, rho, nu,
                    fObj, dfdx, 0.1, 0.5, false)

    xVals = [x[1] for x in xStates]
    yVals = [x[2] for x in xStates]
    plt = plot!(xVals, yVals, markershape = :circle, label = "Iterations $ind")
    ylims!(yMin, yMax)
    xlims!(xMin, xMax)
    display(plt)
    title!("Monte Carlo runs for PD-AL QP Solver")
    savefig("success$ind-PDAL-Up")
end

println("Completed")
