
include("gradDescentLineSearch.jl")
include("backtrackLineSearch.jl")


function logBarrierLine(xInit, fFun, gFunArr, dfdx, dgdxArr, tolx, tolg,
                                    maxIter = 10, verbose = false)
    mu = 1

    veryLargeNum = (1/mu) * 10^10

    numIter = 1
    xError = tolx * 1000

    xNext = xInit
    xPrev = xInit

    lbIterations = [xInit]

    # Modified log to deal with negative inputs
    function modLog(x)
        if x < 0
            return -veryLargeNum
        else
            return log(x)
        end
    end

    # Format the constraints with base fun
    function sumfglog(x, mu)
        val = fFun(x)
        for i in 1:length(x)

            valg = mu * modLog(-gFunArr[i](x))
            val -= valg

            if verbose
                print("Log Constraint $i: ")
                println(valg)
            end
        end
        return val
    end

    # Format the derivative of the constraints
    function sumdfgdxlog(x, mu)
        val = dfdx(x)
#         print("Negative Gradient from Function: ")
#         println(val)
        for i in 1:length(x)

            valg = gFunArr[i](x)

            if valg != 0
                valThisConstraint = -mu * dgdxArr[i](x) / valg
                val += valThisConstraint
            else
                valThisConstraint = [-veryLargeNum, -veryLargeNum]
                val += valThisConstraint
            end

#             print("Negative Gradient from Contraint $i: ")
#             println(valThisConstraint)
        end

#         print("Negative Gradient Total: ")
#         println(val)

        return val
    end

    condEnd = false

    while !condEnd
        if verbose
            println("On iteration $numIter")
        end

        xPrev = xNext

        if verbose && (numIter == 1)
            print("Test Base Function: ")
            println(fFun(xNext))

            print("Test Objective Function: ")
            println(sumfglog(xNext, mu))

            print("Test Base Function Gradient: ")
            println(dfdx(xNext))

            print("Test Objective Function Gradient: ")
            println(sumdfgdxlog(xNext, mu))
        end

        println("--------------")
        print("Test Objective Function: ")
        println(sumfglog(xNext, mu))

        print("Test Objective Function Gradient: ")
        println(sumdfgdxlog(xNext, mu))
        println("--------------")

#         gradAdaptToler2Var(xInit, funcX, gradX, stepInit,
#                     tolerance = 0.5, maxIter = 10, verbose = false)

        useCustomGradDesc = true

        if useCustomGradDesc
            # Using Gradient Descent
            gDIterMax = 10

            alphaStepStart = 0.2

            # gradDescentLineSearch(xInit, funcX, gradX, stepInit,
                    # tolerance = 0.5, maxIter = 10, verbose = false)
            iterations, steps = gradDescentLineSearch(xNext,
                    x -> sumfglog(x, mu), x -> sumdfgdxlog(x, mu),
                    alphaStepStart, tolx * 10, gDIterMax, false)
            xNext = iterations[length(iterations)]
        else

            # Using COTS solution
            result = optimize(x -> sumfglog(x, mu), xNext, BFGS())
            xNext = result.minimizer
        end

        push!(lbIterations, xNext)

        # ------------------
        # Update
        # ------------------
        mu = mu / 10
        xError = norm(xNext - xPrev, 2)

        if verbose
            print("Next step: ")
            println(xNext)
            print("Error: ")
            println(xError)
        end

        # ------------------
        # End Criteria
        # ------------------

        if xError < tolx
            println("x Error Met at $xError")
            condEnd = true
            for i in 1:length(gFunArr)
                errG = max(gFunArr[i](xNext), 0)
                println("g$i Error Met at $errG")
                met = (errG <= tolg)
                condEnd = condEnd && met
            end
        end

        # Prevent the while loop going forever
        # in case there is no convergence.
        if numIter >= maxIter
            condEnd = true
        else
            numIter += 1
        end
    end

    println("Completed in $numIter Iterations")
    return lbIterations

end
