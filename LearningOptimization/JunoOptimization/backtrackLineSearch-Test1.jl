using LinearAlgebra


verbose = false

function backtrackLineSearch(xInit, dirΔ, f, dfdx, paramA, paramB)
    # Output an updated x

    # First, initializae α = 1. THis is the "learning rate"

    α = 1

    # Check the condition:
    # f(x + alpha * dir) > f(x) + a * grad f of (alpha * delta)
    while f(xInit - α * dfdx(xInit)) > f(xInit) - (α/2) * norm(dfdx(xInit), 2)^2
        # If the condition passes, set α to b * α
        if verbose
            print("Left side: ")
            println(f(xInit - α * dfdx(xInit)))
            print("Right side: ")
            print(f(xInit))
            print(" - ")
            print((α/2) * norm(dfdx(xInit), 2)^2)
            print(" = ")
            println(f(xInit) - (α/2) * norm(dfdx(xInit), 2)^2)
        end

        α = paramB * α

        if verbose
            println(α)
        end
    end

    # the updated x is xInit - alpha * gradf(xInit)
    xUpdated = xInit - α * dfdx(xInit)
    # Return updated x
    return xUpdated
end


runTest = false

if runTest
    # Test Script
    xInit1 = 600
    dir = -1
    fFun(x) = x^2
    dfdx(x) = 2 * x
    aTest = 0.3
    bTest = 0.707

    xNew = backtrackLineSearch(xInit1, dir, fFun, dfdx, aTest, bTest)
    println("New x = $xNew")
    println("Result should by 0.181")
end
