# Gradient Descent Code
# Now with line searching

include("backtrackLineSearch.jl")


function gradDescentLineSearch(xInit, funcX, gradX, stepInit,
        tolerance = 0.5, maxIter = 10, verbose = false)

    # xInit is the initial point. Set up initial and error
    xNext = xInit
    xPrev = xInit
    @assert tolerance > 0
    xError = tolerance * 1000

    # Line Search Parameters
    paramA = 0.05 # should be between 0.01 and 0.3
    paramB = 0.4 # should be between 0.1 and 0.8

    # Save key data
    iterations = []
    steps = []
    push!(iterations, xInit)
    push!(steps, stepInit)

    stepNext = stepInit

    numIter = 1 # Count the number of iterations to prevent stalling

    while xError > tolerance

        if verbose
            println("Steping at $stepNext")
        end

        # ------------------
        # Begin Main part of Algorithm
        # ------------------

        if verbose
            println("Updating x-position")
        end
        # Calcuclate the next step direction
        g = gradX(xPrev)
        # dir = - stepInit * g  / norm(g, 2)
        dir = - stepNext * g

        if verbose
            println("Direction: $dir")
        end

        # backtrackLineSearch(xInit, dirΔ, f, dfdx, paramA, paramB, verbose)
        xNext, stepNext = backtrackLineSearch(xPrev, dir, funcX, gradX,
                                    paramA, paramB, verbose)

        # Display Chosen Step Sizing
        if verbose
            print("Moved to ")
            print(funcX(xNext))
            print(" at ")
            println(xNext)
        end

        # ------------------
        # Save the iteration data
        # ------------------

        push!(iterations, xNext)
        push!(steps, stepNext)

        # ------------------
        # End Criteria
        # ------------------

        xError = norm(xNext - xPrev, 2)
        xPrev = xNext

        # Prevent the while loop going forever
        # in case there is no convergence.
        if numIter >= maxIter
           break
        else
            numIter += 1
        end
    end

    println("Finished in $numIter of $maxIter")
#     println(steps)

    return iterations, steps

end




runTest = false

if runTest
    # gradDescentLineSearch(xInit, funcX, gradX, stepInit,
            # tolerance = 0.5, maxIter = 10, verbose = false)

    xInit1 = 600
    fFun(x) = x^2
    dfdx(x) = 2 * x
    stepInit = 10
    tolerance = 0.01
    maxNumIter = 100

    iterations, steps = gradDescentLineSearch(xInit1, fFun, dfdx,
                                        stepInit, tolerance, maxNumIter, true)
    println(iterations)

    using Plots

    plotly()

    scatter(iterations, fFun)
end
