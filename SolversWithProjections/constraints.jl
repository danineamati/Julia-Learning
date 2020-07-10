

# Constraint Types and settings

include("feasibleCheck.jl")
include("projections.jl")

# Augmented Lagrangian Constraints

# ---------------------
# Equality constraints
# ---------------------
struct AL_AffineEquality
    A
    b
end

# Check that a constraint is satisfied (Ax - b = 0)
satisfied(r::AL_AffineEquality, x) = (r.A * x == r.b)
whichSatisfied(r::AL_AffineEquality, x) = (r.A * x .== r.b)

# Evaluate the constraint.
# Raw = Evaluate the function without projection
getRaw(r::AL_AffineEquality, x) = r.A * x - r.b
function getProjVecs(r::AL_AffineEquality, x)
    #=
    We want to project each constraint
    We write Ax = b
    But this is equivalent to a1'x = b1, a2'x = b2, ..., am'x = bm
    Where ak is the row k in A
    =#
    projVecs = []
    for ind in 1:size(r.b, 1)
        aI = r.A[ind, :]
        bI = r.b[ind]
        push!(projVecs, projAffineEq(aI, bI, x))
    end

    return projVecs
end

function getNormToProjVals(r::AL_AffineEquality, x)
    #=
    We want to get the projected vector and calculate the distance between the
    original point and the constraint.
    =#
    projVecs = getProjVecs(r, x)
    pvsDiff = [pv - x for pv in projVecs]
    return norm.(pvsDiff, 2)
end

# Lastly, we do some calculus
function getGradC(r::AL_AffineEquality)
    #=
    We want the gradient of the constraints. This is piecewise in the
    inequality case. But it is a single function in the equality case
    =#
    return r.A
end

function getHessC(r::AL_AffineEquality)
    #=
    We want the hessian of the constraints. This is just zero for affine
    constraints, but we need to match the dimensions.
    =#
    return zeros(size(r.A, 2), size(r.A, 2)) # nxn
end

# -----------------------
# Inequality constraints
# -----------------------
struct AL_AffineInequality
    A
    b
end

# Check that a constraint is satisfied (Ax - b ≤ 0)
satisfied(r::AL_AffineInequality, x) = isFeasiblePolyHedron(r.A, r.b, x)
whichSatisfied(r::AL_AffineInequality, x) = (r.A * x .≤ r.b)

# Evaluate the constraint.
# Raw = Evaluate the function without projection
getRaw(r::AL_AffineInequality, x) = r.A * x - r.b
function getProjVecs(r::AL_AffineInequality, x)
    #=
    We want to project each constraint
    We write Ax = b
    But this is equivalent to a1'x = b1, a2'x = b2, ..., am'x = bm
    Where ak is the row k in A
    =#
    projVecs = []
    for ind in 1:size(r.b, 1)
        aI = r.A[ind, :]
        bI = r.b[ind]
        push!(projVecs, projAffineIneq(aI, bI, x))
    end

    return projVecs
end

function getNormToProjVals(r::AL_AffineInequality, x)
    #=
    We want to get the projected vector and calculate the distance between the
    original point and the constraint.
    =#
    projVecs = getProjVecs(r, x)
    pvsDiff = [pv - x for pv in projVecs]
    return norm.(pvsDiff, 2)
end

# Lastly, we do some calculus
function getGradC(r::AL_AffineInequality, x)
    #=
    We want the gradient of the constraints. This is piecewise in the
    inequality case. But it is a single function in the equality case
    =#
    APost = zeros(size(r.A))
    rowPassed = whichSatisfied(r, x)
    for row in 1:size(r.A, 1)
        if !rowPassed[row]
            # Constraint is not met
            APost[row, :] = r.A[row, :]
        end
    end
    return APost
end

function getHessC(r::AL_AffineInequality)
    #=
    We want the hessian of the constraints. This is just zero for affine
    constraints, but we need to match the dimensions.
    =#
    return zeros(size(r.A, 2), size(r.A, 2)) # nxn
end


# -----------------------
# Second-Order Cone Constraints
# -----------------------
# Specifically has the form ||Ax - b||_p ≤ c'x - d
# Where p denotes which norm (usually p = 2)

struct AL_pCone
    A
    b
    c
    d
    p
end

# Evaluate the constraint.
# Raw = Evaluate the function without projection
getRaw(r::AL_pCone, x) = norm(r.A * x - r.b, r.p) - r.c' * x + r.d

# Check that a constraint is satisfied (||Ax - b|| ≤ c'x - d)
satisfied(r::AL_pCone, x) = (getRaw(r, x) ≤ 0)
whichSatisfied(r::AL_pCone, x) = (getRaw(r, x) .≤ 0)


function getProjVecs(r::AL_pCone, x, verbose = false)
    #=
    We want to project onto the cone where
    v = ||Ax - b||
    s = cx - d
    =#
    v = r.A * x - r.b
    s = r.c' * x - r.d

    proj = projSecondOrderCone(v, s, r.p)

    if verbose
        println("x = $x -> Inside? $(satisfied(r, x))")
        println("v = $v, s = $s")
        println("proj = $proj")
    end

    vproj = proj[1:end - 1]
    xproj = r.A \ (r.b + vproj)

    if verbose
        println("vproj = $vproj -> $(norm(vproj, r.p))")
        print("xproj = $xproj -> $(r.A * xproj - r.b) -> ")
        print("$(norm(r.A * xproj - r.b, r.p))")
        println(" vs $(r.c' * xproj - r.d) vs $(proj[end])")
        println("Satisfied? $(satisfied(r, xproj))")
        println()
    end
    return xproj, proj
end

function getNormToProjVals(r::AL_pCone, x)
    #=
    We want to get the projected vector and calculate the distance between the
    original point and the constraint.
    =#
    projVec = getProjVecs(r, x)
    projDiff = projVec[1] - x
    return norm(projDiff, 2)
end

# Lastly, we do some calculus
function getGradC(r::AL_pCone, x, verbose = false)
    #=
    We want the gradient of the constraints. This is piecewise due to the
    inequality case. This assumes a 2-norm (for now)
    =#
    if satisfied(r, x)
        if verbose
            println("Satisfied")
        end
        return zeros(size(r.A, 2))
    else
        if verbose
            println("NOT Satisfied")
        end
        num = (r.A)' * (r.A * x - r.b)
        den = norm(r.A * x - r.b, 2) # Formula is not for a general norm yet
        return (num/den) - r.c
    end
end

function getHessC(r::AL_pCone, x)
    #=
    We want the hessian of the constraints. This is just zero for affine
    constraints, but nonzero for second order cone constraints.

    Again, this assumes a 2-norm for now.
    =#
    if satisfied(r, x)
        return zeros(size(r.A, 2), size(r.A, 2))
    end

    normVal = norm(r.A * x - r.b, 2)
    numGrad = (r.A)' * (r.A * x - r.b)

    term1 = ((r.A)' * r.A) / normVal  # Must be nxn
    term2 = (numGrad * numGrad') / (normVal^2) # Must be nxn
    return term1 + term2
end

# -----------------------
# Second-Order Cone Constraints
# -----------------------
#=
Specifically has the form:
||s|| ≤ t
Ax - b = s
c'x - d = t

So,
||s|| - t ≤ 0
Ax - b - s = 0
c'x - d - t = 0

I use the second order cone
=#


struct AL_coneSlack
    A
    b
    c
    d
end

# Evaluate the constraint.
# Raw = Evaluate the function without projection
getRaw(r::AL_coneSlack, x, s, t) = [norm(s, 2) - t;
                                    r.A * x - r.b - s;
                                    r.c'x - r.d - t]

# Check that a constraint is satisfied (||Ax - b|| ≤ c'x - d)
function whichSatisfied(r::AL_coneSlack, x, s, t)
    raw = getRaw(r, x, s, t)
    return [raw[1] ≤ 0; raw[2:end] .== 0]
end
function satisfied(r::AL_coneSlack, x, s, t)
    wSat = whichSatisfied(r, x, s, t)
    for iw in wSat
        if !iw
            # A single entry is false
            return false
        end
    end
    # All entries are true
    return true
end



function getProjVecs(r::AL_coneSlack, x, s, t, verbose = false)
    #=
    We want to get each of the projection vectors.
    For the cone, we have the (s, t) cone.

    For the equality constraints we have the affine equality projections
    Ax = b + s
    But this is equivalent to
    a1'x = b1 + s1, a2'x = b2 + s2, ..., am'x = bm + sm
    Where ak is the row k in A

    For the last equality constraint, we simply have c'x = d + t
    =#
    projVecs = []

    # Cone Constraints
    coneproj = projSecondOrderCone(s, t)

    if verbose
        println("Cone Projection: $coneproj")
    end

    push!(projVecs, coneproj)

    # Set of Equality Constraints
    for ind in 1:size(r.b, 1)
        aI = r.A[ind, :]
        bI = r.b[ind]
        sI = s[ind]
        push!(projVecs, projAffineEq(aI, bI + sI, x))
    end

    # Last Equality Constraint
    push!(projVecs, projAffineEq(r.c, r.d + t, x))

    return projVecs
end


function getNormToProjVals(r::AL_coneSlack, x, s, t)
    #=
    We want to get the projected vector and calculate the distance between the
    original point and the constraint.
    =#
    projVec = getProjVecs(r, x, s, t)
    coneDiff = norm([s; t] - projVec[1])
    projDiff = [norm(pv - x, 2) for pv in projVec[2:end]]
    return [coneDiff; projDiff]
end

# Lastly, we do some calculus
function getGradC(r::AL_coneSlack, x, s, t, verbose = false)
    #=
    We want the gradient of the constraints. This is piecewise due to the
    inequality case. This assumes a 2-norm (for now).

    When the constraint is violated, we have:
    J'(ρ c + λ)
    Where J is the jacobian of c. In particular
    J = [ 0(1xn) s'/||s||    -1 ]
        [ A      -1 * I      0  ]
        [ c'       0         -1 ]

    This function will return J
    =#

    sizeJcols = size(x, 1) + size(s, 1) + size(t, 1)
    sizeJrows = size(x, 2) + size(s, 1) + size(t, 2)

    if satisfied(r, x, s, t)
        if verbose
            println("Satisfied")
        end
        return zeros(sizeJcols)
    else
        if verbose
            println("NOT Satisfied")
        end
        jacobRow1 = [zeros(size(x')) s'/norm(s,2) -1]
        if verbose
            println("Row 1")
            display(jacobRow1)
        end
        jacobRow2 = [r.A (Diagonal(-1*ones(size(r.A, 1)))) zeros(size(r.A, 1))]
        if verbose
            println("Row 2")
            display(reshape(jacobRow2, size(r.A, 1), sizeJcols))
        end
        jacobRow3 = [r.c' zeros(1, size(r.A, 1)) -1]
        if verbose
            println("Row 3")
            display(jacobRow3)
        end
        jacob = [jacobRow1; jacobRow2; jacobRow3]
        if verbose
            println("Size expected = $sizeJrows x $sizeJcols ?= $(size(jacob))")
            display(reshape(jacob, size(jacob)))
            # display(reshape(jacob, sizeJrows, sizeJcols))
        end

        return jacob
    end
end

function getHessC(r::AL_coneSlack, x)
    #=
    We want the hessian of the constraints. This is just zero for affine
    constraints, but nonzero for second order cone constraints.

    Again, this assumes a 2-norm for now.
    =#
    if satisfied(r, x)
        return zeros(size(r.A, 2), size(r.A, 2))
    end

    normVal = norm(r.A * x - r.b, 2)
    numGrad = (r.A)' * (r.A * x - r.b)

    term1 = ((r.A)' * r.A) / normVal  # Must be nxn
    term2 = (numGrad * numGrad') / (normVal^2) # Must be nxn
    return term1 + term2
end

runTestsAffine = false
runTestsOldCone = false
runTestsNewCone = true

if runTestsAffine
    println()
    println("Affine Equality Constraints:")
    dT = AL_AffineEquality([5], [10])
    println("** Simplest")
    println("On line  (True) : $(satisfied(dT, 2))")
    println("Off line (False): $(satisfied(dT, 4)) and $(satisfied(dT, 0))")
    dT2 = AL_AffineEquality([5 0; 0 6], [10; 12])
    println("** 2D")
    println("On Intesection (True)        : $(satisfied(dT2, [2; 2]))")
    println("Check which not (True, False): $(whichSatisfied(dT2, [2; 4]))")
    println("** Testing the projections and violations")
    testpt = [10; 12]
    println("Point $testpt not at intersetion : $(!satisfied(dT2, testpt))")
    println("Projection onto constraints : $(getProjVecs(dT2, testpt))")
    gN = getNormToProjVals(dT2, testpt)
    println("Violation is : $gN")
    println("Total Violation is : $(gN'gN)")
    println("** Testing Calculus")
    println("Grad = $(getGradC(dT2)) and Hessian = $(getHessC(dT2))")

    println("\nAffine Inequality Constraints")
    diT = AL_AffineInequality([5], [10])
    println("** Simplest")
    println("On line    (True) : $(satisfied(diT, 2))")
    println("Above line (False): $(satisfied(diT, 4))")
    println("Under line (True) : $(satisfied(diT, 0))")
    diT2 = AL_AffineInequality([5 0; 0 6], [10; 12])
    println("** 2D")
    println("On Intesection (True)        : $(satisfied(diT2, [2; 2]))")
    println("Check which not (True, False): $(whichSatisfied(diT2, [2; 4]))")
    println("Check which not (True, True) : $(whichSatisfied(diT2, [2; 0]))")
    println("** Testing the projections and violations")
    testpt = [10; 12]
    println("Point $testpt not at in region : $(!satisfied(diT2, testpt))")
    println("Projection onto feasible region : $(getProjVecs(diT2, testpt))")
    gN = getNormToProjVals(diT2, testpt)
    println("Violation is : $gN")
    println("Total Violation is : $(gN'gN)")
    testpt2 = [-10; -12]
    println("Point $testpt2 in feasible region : $(satisfied(diT2, testpt2))")
    println("Projection onto feasible region : $(getProjVecs(diT2, testpt2))")
    gN2 = getNormToProjVals(diT2, testpt2)
    println("Violation is : $gN2")
    println("Total Violation is : $(gN2'gN2)")
    println("** Testing Calculus")
    println("Hessian = $(getHessC(diT2))")
    println("Grad at $testpt = $(getGradC(diT2, testpt))")
    println("Grad at $testpt2 = $(getGradC(diT2, testpt2))")
end

if runTestsOldCone
    println("\nSecond-Order Constraints")
    dconeT = AL_pCone([5][:,:], vcat([-4]), vcat([3]), -8, 2)
    xRan = -5:5
    println("\n** Simplest Full")
    println("xVals = $(collect(xRan))")
    println("Raw Vals = $([getRaw(dconeT, vcat([x])) for x in xRan])")
    println("Satisfied = $([satisfied(dconeT, vcat([x])) for x in xRan])")
    println("Projection = $([getProjVecs(dconeT, vcat([x])) for x in xRan])")
    println("Violation = $([getNormToProjVals(dconeT, vcat([x])) for x in xRan])")

    dconeT2D = AL_pCone([5 3; 3 4], [-4; -2], [3; 4], -8, 2)
    xRan = -4:5
    println("\n** 2D Full")
    println("xVals = $(collect(xRan))")
    println("In the form [x; x]")
    println("Raw Vals = $([getRaw(dconeT2D, [x; x]) for x in xRan])")
    println("Satisfied = $([satisfied(dconeT2D, [x; x]) for x in xRan])")
    projVecList = [getProjVecs(dconeT2D, [x; x], true) for x in xRan]
    println("Projection = ")
    display(projVecList)
    println("Violation = $([getNormToProjVals(dconeT2D, [x; x]) for x in xRan])")
    println("Gradients = ")
    gradVecs = [getGradC(dconeT2D, [x; x]) for x in xRan]
    display(gradVecs)
    println("Hessians = ")
    hessVecs = [getHessC(dconeT2D, [x; x]) for x in xRan]
    display(hessVecs)

    # dconeT2DSimp = AL_pCone([2 1/3; 1/3 3/4], [0; 0], [0; 0], -8, 2)
    # xRan = -4:5
    # println("\n** 2D Simple (no b, no c)")
    # println("xVals = $(collect(xRan))")
    # println("In the form [x; x]")
    # println("Raw Vals = $([getRaw(dconeT2DSimp, [x; x]) for x in xRan])")
    # println("Satisfied = $([satisfied(dconeT2DSimp, [x; x]) for x in xRan])")
    # projVecList = [getProjVecs(dconeT2DSimp, [x; x], true) for x in xRan]
    # println("Projection = ")
    # display(projVecList)
    # print("Violation = ")
    # println("$([getNormToProjVals(dconeT2DSimp, [x; x]) for x in xRan])")
end

if runTestsNewCone
    dslackCone1 = AL_coneSlack([2 1/3; 1/3 3/4; 1 1], [3; 4; 5], [2; 1], -8)

    println("\n** Cone With Slack Variables")
    display(dslackCone1)
    println("Raw Values: ")
    xRan = -1:1
    s = [2; 5; -4]
    t = 5
    println([getRaw(dslackCone1, [x; x], s, t) for x in xRan])
    println("Satisfied: ")
    println([whichSatisfied(dslackCone1, [x; x], s, t) for x in xRan])
    println("Projections: ")
    println([getProjVecs(dslackCone1, [x; x], s, t) for x in xRan])
    println("Violations: ")
    println([getNormToProjVals(dslackCone1, [x; x], s, t) for x in xRan])
    println("Gradient")
    jc = getGradC(dslackCone1, [-2; -2], s, t)
    display(reshape(jc, size(jc)))
end
