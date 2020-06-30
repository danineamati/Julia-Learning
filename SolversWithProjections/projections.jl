

# Covers how to consider a square violation
function sqrDistEuclid(pt, projFunc)
    #=
    Choose a given projection method and determine the squared distance
    at this point.
    =#
    return norm(pt - projFunc(pt), 2)^2
end



# Covers Projections

# First, Lines from vectors at the origin
function projLine(pt, vec)
    #=
    This function projects a point pt onto a line described by vec

    proj_v(p) = v * (p'v)/(v'v)
    =#
    return vec * (pt'vec) / (vec'vec)
end

function orthoProjLine(pt, vec)
    #=
    This function projects a point pt onto the orthogonal of a line described
    by vec

    proj_v(p) = v * (p'v)/(v'v)
    proj_v⟂(p) = pt - v * (p'v)/(v'v)
    =#
    return pt - projLine(pt, vec)
end

# Second, Positive Orthant
function projPosOrth(pt)
    #=
    This function is for a constraint of the form "τ ≥ 0."
    When τ_i ≥ 0, the variable τ is in the positive orthant, thus, the
    projection is simply τ_i.
    When τ_i < 0, the variable τ is not in the positive orthant, thus, the
    projection is distance to the positive orthant
    =#

    return max.(pt, 0)
end

# Third, Affine Case
function projAffineEq(a, b, x)
    #=
    Projection for Affine Equality Case (a'x = b)

    We can show that the projection is given by
    proj(x) = x + (b - a'x) * a / a'a
            = x + a * [b] / a'a - a * (a'x) / a'a
    =#
    t1 = a * b / a'a
    t2 = a * (a'x) / a'a
    return x + t1 - t2
end


function projAffineIneq(a, b, x)
    #=
    Projection for Affine Inequality Case (a'x ≤ b)

    We can show that the projection is given by
    projAffineEq(x) for a'x > b
    x               for a'x ≤ b
    =#
    if a'x > b
        return projAffineEq(a, b, x)
    end
    return x
end

function projSecondOrderCone(v, s)
    #=
    Projection for Second-Order Cone (AKA quadratic cone or the Lorentz cone)

    The second-order cone is C = {(x, t) ∈ R^n+1 | ||x||2 ≤ t}. Using the
    2-norm. Projection onto it is given by

    proj(v, s) =
    0                                   for ||v|| ≤ -s  (Below the tip)
    (v, s)                              for ||v|| ≤ s   (In the cone)
    (1/2)(1 + s/||v||)(v, ||v||)        for ||v|| ≥ |s| (Onto the cone)

    Note that (|s| = absolute value of s)
    =#

    if norm(v, 2) ≤ -s
        return zeros(size([v; s]))
    elseif norm(v, 2) ≤ s
        return [v; s]
    elseif norm(v, 2) ≥ s
        return (1/2) * (1 + s / norm(v, 2)) * [v; norm(v, 2)]
    end

    println("Second Order Cone Conditions ERROR")
    return -1

end



runTests = false

if runTests
    println("\n")
    print("Testing projection onto a line: ")
    vTest = [1; 3]
    ptTest = [0; -2]
    projptTest = projLine(ptTest, vTest)
    print("$projptTest ?= [-0.6; -1.8] : Passed? ")
    println(" $(projptTest == [-0.6; -1.8])")
    optTest = round.(orthoProjLine(ptTest, vTest), digits=5)
    print("Orthogonal projection onto a line: ")
    print("$optTest ?= [0.6; -0.2] : Passed? ")
    println(" $(optTest == [0.6; -0.2])")

    println()
    print("Now we try a projection onto the positive orthant: ")
    pt2Test = [5; -3; -100; 0]
    projpt2Test = projPosOrth(pt2Test)
    println("$pt2Test -> $projpt2Test : Passed? $(projpt2Test .≥ 0)")

    println()
    print("Next up is an affine constraint: ")
    aTest = [5; 4; 3]
    bTest = 4
    pt3Test = [10; 10; 17]
    projpt3Test = projAffineEq(aTest, bTest, pt3Test)
    xp = pt3Test - projpt3Test
    println("$pt3Test -> $projpt3Test")
    print("On Hyperplane : $(aTest'projpt3Test == bTest), ")
    println("Min Dist : $(xp'aTest == norm(xp) * norm(aTest))")

    println()
    println("Lastly is the second order cone")
    println("The first cone is the simplest unit cone at the origin")
    sTest1 = 6
    pt4Test = [3; 4; 5]
    projpt4Test = projSecondOrderCone(pt4Test, sTest1)
    print("$pt4Test, $sTest1 -> $projpt4Test, ")
    println("Base norm = $(norm(pt4Test))")
    sTest2 = 10
    projpt4Test = projSecondOrderCone(pt4Test, sTest2)
    print("$pt4Test, $sTest1 -> $projpt4Test, ")
    println("Base norm = $(norm(pt4Test))")
    sTest3 = -10
    projpt4Test = projSecondOrderCone(pt4Test, sTest2)
    print("$pt4Test, $sTest1 -> $projpt4Test, ")
    println("Base norm = $(norm(pt4Test))")

    print("The second cone now is offeset with v -> Ax - b -> ")
    AMat = [4 5; -4 5; 0 -1]
    bVec = [20; 30; 1]
    xTest = [3; 4]
    sTest2 = 10
    projpt5Test = projSecondOrderCone(AMat * xTest - bVec, sTest2)
    println(" $(AMat * xTest - bVec) for x = $xTest")
    println("Proj -> $projpt5Test")

end
