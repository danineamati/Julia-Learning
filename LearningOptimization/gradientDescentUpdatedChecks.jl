# Gradient Descent Code
# Now with line searching


function gradDescentLineSearch(xInit, funcX, gradX, stepInit,
        tolerance = 0.5, maxIter = 10, verbose = false)

    # xInit is the initial point. Set up initial and error
    xNext = xInit
    xPrev = xInit
    @assert tolerance > 0
    xError = tolerance * 1000

    # Step size
    stepNext = stepInit
    stepUpFactor = 1.2
    stepDownFactor = 0.5

    # Save key data
    iterations = []
    steps = []
    push!(iterations, xInit)
    push!(steps, stepInit)

    numIter = 1

    while xError > tolerance

        if verbose
            println("Steping at $stepNext")
        end

        # ------------------
        # Begin Main part of Algorithm
        # ------------------
        if verbose
            println("Calculating the gradient")
        end
        # Calculate the gradient
        g = gradX(xNext)
        if verbose
            println("Updating x-position")
        end
        # Calcuclate the next step
        xNextInConsideration = xPrev - stepNext * g / norm(g, 2)

        # Adaptive Step Sizing
        if verbose
            print("Moved to ")
            print(funcX(xNextInConsideration))
            print(" at ")
            println(xNextInConsideration)
        end

        if funcX(xNextInConsideration) <= funcX(xPrev)
            # Moved towards the minimum

            # Increase Stepsize
            stepNext = stepUpFactor * stepNext

            # Accept Step
            xNext = xNextInConsideration

            # Update the error
            xError = norm(xPrev - xNext, 2)

            xPrev = xNext

            if verbose
                println("Accepted")
            end
        else
            # Decrease Stepsize
            stepNext = stepDownFactor * stepNext

            # Reject Step
            if verbose
                println("*** Rejected ***")
            end
        end

        # ------------------
        # Save the iteration data
        # ------------------

        push!(iterations, xNext)
        push!(steps, stepNext)

        # ------------------
        # End Criteria
        # ------------------

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




runTest = true

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
                                        stepInit, tolerance, maxNumIter, false)
    println(iterations)

    using Plots

    plotly()

    scatter(iterations, fFun)
end
