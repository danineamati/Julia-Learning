

# Covers Projections

function projLine(pt, vec)
    #=
    This function projects a point pt onto a line described by vec

    proj_v(p) = v * (p.v)/(v.v)
    =#
    return vec * (pt'vec) / (vec'vec)
end

function orthoProjLine(pt, vec)
    #=
    This function projects a point pt onto the orthogonal of a line described
    by vec

    proj_v(p) = v * (p.v)/(v.v)
    proj_v⟂(p) = pt - v * (p.v)/(v.v)
    =#
    return pt - projLine(pt, vec)
end


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


# Projection for Affine Equality Case (Ax = b)


# Projection for Affine Inequality Case (Ax ≤ b)


# Projection for Second-Order Cone



runTests = true

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

end
