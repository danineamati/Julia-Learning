# Test the Newton Update part of the Primal-Dual Scheme

include("primal-dual-IP-Newton-QP.jl")


println()
println("Testing Newton Step on test matrix")
matA = [4 5 9; 3 2 1; 0 9 10]
matB = [2 4 5; 8 6 7; 1 4 2]
matC = [8 5 7; 3 1 6; 4 9 2]
matD = [4 5 2; 8 6 4; 1 2 3]

rVec = [3; 4; 5; 9; 8; 7]

nextStep = newtonStep(matA, matB, matC, matD, rVec)
display(nextStep)

println()
println("Setting Up the QP")
QMat, cVec, AMat, bVec, x0 = QPSetup(true, true)

fObj(x) = (1/2) * x'QMat*x + cVec'x
dfdx(x) = QMat * x + cVec

println("Objective: f(x) = (1/2) x'Qx + c'x")
println("Q = $QMat")
println("c = $cVec")
println("Constraints: Ax ≦ b")
println("A = $AMat")
println("b = $bVec")
println("Initial Starting Point: $x0")

println()
println("Testing r Vec from QP-Setup")
lambda = [1; 1; 1]
mu = 1
rVec = getQPrVec(QMat, cVec, AMat, bVec, x0, lambda, mu)
grA, grB, grC, grD = getQPGradrVec(QMat, AMat, bVec, x0, lambda)
println("r vec = $rVec")
println("∇r =")
display([grA grB; grC grD])

dirNewton = newtonStep(grA, grB, grC, grD, rVec)
println()
println("First Newton Step: ")
display(dirNewton)

# backtrackLineSearch(xInit, dirΔ, f, dfdx, paramA, paramB, verbose = false)
paramA = 0.1
paramB = 0.5
x0LS, stepLS = backtrackLineSearch(x0, dirNewton[1:2], fObj, dfdx,
                                paramA, paramB)
println("Final Deduced Step Direction: $x0LS at α = $stepLS")

# # Update the rVector
# x0New = x0 + x0LS
# lambdaNew = lambda + stepLS * rVec[3:5]
#
# rVec = vcat(x0New, lambdaNew)
#
# grA, grB, grC, grD = getQPGradrVec(QMat, AMat, bVec, x0LS, lambdaNew)
# println("r vec = $rVec")
# println("∇r =")
# display([grA grB; grC grD])
#
# dirNewton = newtonStep(grA, grB, grC, grD, rVec)
# println()
# println("First Newton Step: ")
# display(dirNewton)
# # backtrackLineSearch(xInit, dirΔ, f, dfdx, paramA, paramB, verbose = false)
# paramA = 0.1
# paramB = 0.5
# x0LS2, stepLS2 = backtrackLineSearch(x0New, dirNewton[1:2], fObj, dfdx,
#                                 paramA, paramB)
# println("Final Deduced Step Direction: $x0LS2 at α = $stepLS2")
#
# # Update the rVector
# x0New2 = x0New + x0LS2
# lambdaNew2 = lambdaNew + stepLS2 * rVec[3:5]
#
# itersCoords = [x0, x0New, x0New2]
# xVals = [x[1] for x in itersCoords]
# yVals = [x[2] for x in itersCoords]
#
# xRange = -10:0.1:10
# yMax = 14
# yMin = -3
#
# plt = constraintsPlot(AMat, bVec, x0, xRange, yMin, yMax)
# pltfollow = plot!(xVals, yVals, label = "Newton Steps Path")
# display(pltfollow)

# Putting it all together
println("Putting all together from the start: ")
hCurr = vcat(x0, lambda)

# (Q, c, A, b, h, mu, fObj, dfdx)
mu = 1
hVNew = newtonAndLineSearch(QMat, cVec, AMat, bVec, hCurr, mu, fObj, dfdx)
display(hVNew)
println()
# Putting it all together in loop
println("Putting all together from the start iteratively: ")

hStates = []
push!(hStates, hCurr)

paramA = 0.1
paramB = 0.5

mu = 1
muReduct = 0.1

for i in 1:10
    # Update rVec at each iteration
    global hCurr = newtonAndLineSearch(QMat, cVec, AMat, bVec, hCurr, mu,
                                    fObj, dfdx, paramA, paramB, true)
    push!(hStates, hCurr)
    global mu = mu * muReduct
end

xVals = [h[1] for h in hStates]
yVals = [h[2] for h in hStates]

plot!(xVals, yVals, label = "Iterative")
pltIter = scatter!(xVals, yVals, label = "Iterative")

display(pltIter)

println()
println("Completed")
