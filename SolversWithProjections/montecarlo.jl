

# This function takes care of the Monte Carlo aspects of the tests
using Distributions

function getPoint(minVec, maxVec)
    #=
    returns a random point in the rectangle specified by the min and max
    vectors
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
    range.s
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
