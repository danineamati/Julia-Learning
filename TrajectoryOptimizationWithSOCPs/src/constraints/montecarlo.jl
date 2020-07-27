

# This function takes care of the Monte Carlo aspects of the tests
using Distributions

function getPoint(minVec, maxVec)
    #=
    Returns a random point in the rectangle specified by the min and max
    vectors

    In this case, a uniform distribution is used. In future work, a different
    distribution could be used without loss of generality.
    =#
    pt = zeros(size(minVec))
    for dimen in 1:size(minVec, 1)
        pt[dimen] = rand(Uniform(minVec[dimen], maxVec[dimen]))
    end
    return pt
end

function montecarlo(minVec, maxVec, numPoints = 10, funcCheck = x -> true)
    #=
    This function takes a given range and randomly selects a point in that
    range. It then checks if that point passes a check before adding it.
    The result is a list of random points.

    In general, the check function is for the feasible set.
    Ex: x -> isFeasiblePolyHedron(A1, b1, x)

    If the check function is left as default, all points pass and are added
    to the list.
    =#

    listPoints = []

    while size(listPoints, 1) < numPoints
        newpt = getPoint(minVec, maxVec)

        if funcCheck(newpt)
            push!(listPoints, newpt)
        end
    end

    return listPoints
end

function inBox(x2D)
    if (1 > x2D[1] > 0) && (1 > x2D[2] > 0)
        return true
    end
    return false
end


runTests = false

if runTests
    println("\n")
    print("Generating a 3D point: ")
    pt1 = getPoint([3; 2; -5], [10; 100; 0.5])
    println(pt1)
    print("Checking that the point is in the bounds: ")
    print("$(10 > pt1[1] > 3) & $(100 > pt1[2] > 2)")
    println(" & $(0.5 > pt1[3] > -5)")

    println("Generating 3 Random 4D Points")
    display(montecarlo([3; 2; -5; -0.01], [10; 100; 0.5; 0.01], 3))

    println("Generating 2 Random Points in a smaller [0, 1] box")
    display(montecarlo([-20; -20], [20; 20], 2, inBox))
end
