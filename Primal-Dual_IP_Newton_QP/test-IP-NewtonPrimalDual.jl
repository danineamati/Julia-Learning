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

println("Objective: f(x) = (1/2) x'Qx + c'x")
println("Q = $QMat")
println("c = $cVec")
println("Constraints: Ax ≦ b")
println("A = $AMat")
println("b = $bVec")
println("Initial Starting Point: $x0")

println()
println("Testing r Vec from QP-Setup")
lambda = [1/20; 1/30; 1]
mu = 1
rVec = getQPrVecIP(QMat, cVec, AMat, bVec, x0, lambda, mu)
grA, grB, grC, grD = getQPGradrVecIP(QMat, AMat, bVec, x0, lambda)
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
x0LS, stepLS = backtrackLineSearch(x0, dirNewton[1:2], fObj, dfdx,
                                paramA, paramB)
xDed = stepLS * dirNewton[1:2]
println("Final Deduced Step Direction: $xDed at α = $stepLS")

if true
    # Putting it all together
    println()
    println("Putting all together from the start: ")
    hCurr = vcat(x0, lambda)

    # (Q, c, A, b, h, mu, fObj, dfdx)
    mu = 1
    hVNew = newtonAndLineSearch(QMat, cVec, AMat, bVec, hCurr, mu, fObj, dfdx)
    display(hVNew)
    println()
    # Putting it all together in loop
    println("Putting all together from the start iteratively: ")

    # hStates = []
    # push!(hStates, hCurr)
    #
    # paramA = 0.1
    # paramB = 0.5
    #
    # mu = 1
    # muReduct = 0.1
    #
    # for i in 1:10
    #     # Update rVec at each iteration
    #     global hCurr = newtonAndLineSearch(QMat, cVec, AMat, bVec, hCurr, mu,
    #                                     fObj, dfdx, paramA, paramB, true)
    #     push!(hStates, hCurr)
    #     global mu = mu * muReduct
    # end

    mu = 1
    hStates = pdIPNewtonQPmain(QMat, cVec, AMat, bVec, x0, lambda, mu,
                                    fObj, dfdx)


    xVals = [h[1] for h in hStates]
    yVals = [h[2] for h in hStates]

    if cVec == [4; -3]
        xCorrect = [-2.04348]
        yCorrect = [1.65217]

        scatter!(xCorrect, yCorrect, label = "Minimum", markershape = :xcross,
                    markercolor = :red, markersize = 10)
    end

    plot!(xVals, yVals, label = "Iterative")
    pltIter = scatter!(xVals, yVals, label = "Iterative")
    title!("Primal-Dual Newton Progression")

    display(pltIter)

    println()
end
println("Completed")
