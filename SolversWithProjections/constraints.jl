

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
function coneSatisfied(r::AL_coneSlack, s, t)
    return (norm(s, 2) - t ≤ 0)
end

function coneValOriginal(r::AL_coneSlack, x)
    sInt = norm(r.A * x - r.b, 2)
    tInt = r.c'x - r.d
    return sInt - tInt
end

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

function coneActive(s, t, λ)
    #=
    returns "ACITVE" and "FILLED"

    "ACTIVE" = should return something nonzero
    "FILLED" = should project onto a solid cone, not the boundary

    We want to "activate" this constraint when ||s|| - t > 0 OR λ > 0.
    This yields 4 cases:

    if ||s|| - t > 0:
        OUTSIDE CONE
        if λ > 0:
            ACTIVE
            The solver is compensating for the constraint
        if λ = 0:
            ACTIVE
            The λ has not been initialized, but the constraint is active
    if ||s|| - t ≤ 0:
        INSIDE CONE
        if λ > 0:
            ACTIVE
            Must project to the boundary of the cone
        if λ = 0:
            INACTIVE
            Constraint is fully satisfied
    =#

    cVal = norm(s, 2) - t

    if cVal > 0
        # OUTSIDE CONE and ACTIVe
        active = true
        filled = true
    else
        # INSIDE CONE
        if λ > 0
            # ACTIVE
            println("ACTIVE & NOT FILLED")
            active = true
            filled = false
        else
            # INACTIVE
            active = false
            filled = true
        end
    end

    return active, filled

end

function getProjVecs(r::AL_coneSlack, x, s, t, filled = true, verbose = false)
    #=
    We want to get each of the projection vectors.
    For the cone, we have the (s, t) cone given by ||s|| ≤ t.
    We activate this based on the function above

    For the equality constraints we have the affine equality projections
    Ax = b + s
    But this is equivalent to
    a1'x = b1 + s1, a2'x = b2 + s2, ..., am'x = bm + sm
    Where ak is the row k in A

    For the last equality constraint, we simply have c'x = d + t
    =#
    projVecs = []
    signs = []

    # Cone Constraints
    coneproj = projSecondOrderCone(s, t, filled)

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
        if (aI'x - bI - sI) ≥ 0
            push!(signs, 1)
        else
            push!(signs, -1)
        end
    end

    # Last Equality Constraint
    push!(projVecs, projAffineEq(r.c, r.d + t, x))
    if (r.c'x - r.d - t) ≥ 0
        push!(signs, 1)
    else
        push!(signs, -1)
    end

    return projVecs, signs
end


function getNormToProjVals(r::AL_coneSlack, x, s, t, λ = 0)
    #=
    We want to get the projected vector and calculate the distance between the
    original point and the constraint.
    =#
    active, filled = coneActive(s, t, λ)
    projVec, signs = getProjVecs(r, x, s, t, filled)
    coneDiff = norm([s; t] - projVec[1], 2)
    if !filled
        coneDiff *= -1
    end
    projDiff = [sign * norm(pv - x, 2) for (sign, pv) in
                                            zip(signs, projVec[2:end])]
    return [coneDiff; projDiff]
end

# Lastly, we do some calculus
function getGradC(r::AL_coneSlack, x, s, t, verbose = false)
    #=
    We want the gradient of the constraints. This is piecewise due to the
    inequality case. This assumes a 2-norm (for now).

    First, let's consider the inequality constraints.
    → [ρ I_λ c(y) + λ]'∇c(y)

    where I_λ is 1 if [λ > 0 OR g(y) > 0] and 0 otherwise.
    ∇c(y) = [0; s/||s||; -1]

    Second, let's consider the equality constraints.
    → J(h(y))'(ρ h(y) + κ)
            x       s         t
    J = [ A      -1 * I       0  ]  h1
        [ c'       0         -1  ]  h2

    If we put these together, we get
            x       s         t
        [0       s/||s||     -1  ]  c1
    J = [ A      -1 * I       0  ]  h1
        [ c'       0         -1  ]  h2
    =#

    sizeJcols = size(x, 1) + size(s, 1) + size(t, 1)
    sizeJrows = size(t, 2) + size(s, 1) + size(t, 2)

    # if satisfied(r, x, s, t)
    #     if verbose
    #         println("Satisfied")
    #     end
    #     return zeros(sizeJrows, sizeJcols)

    # if coneSatisfied(r, s, t)
    #     jacobRow1 = zeros(1, sizeJcols)
    #     if verbose
    #         println("Cone Satisfied")
    #     end
    # else
    #     jacobRow1 = [zeros(size(x')) s'/norm(s,2) -1]
    # end

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

function getHessC(r::AL_coneSlack, x, s, t, λCone = 0)
    #=
    We want the hessian of the combined constraint terms. Namely:
    ρc(x)'c(x) + λ c(x)

    Where
            [ ||s|| - t ]
    c(x) =  [ Ax - b - s]
            [c'x - d - t]

    Again, this assumes a 2-norm for now.

    We have three primal variables (x, s, t). Thus the Hessian is symmetric
    with this (n+m+1)x(n+m+1) dimensionality.

    We separate the Hessian into 6 submatrices
        [A  B  C]
    H = [B' D  E]
        [C' E' F]

    Focusing first on the cone inequality constraint
            n               m                    1
        [0              0                     0        ]   n
    H = [0              ss'/||s||)            -s/||s|| ]   m
        [0              -s'/||s||             1        ]   1

    When [λ > 0 OR g(y) > 0], otherwise it is uniformly zero.

    Then, we focus on the equality constraints
            n               m           1
        [A'A + cc'      -A'         -c      ]   n
    H = [-A             I_m         0       ]   m
        [-c'            0           1       ]   1

    =#

    sizeH = size(x, 1) + size(s, 1) + size(t, 1)

    # Equality Constraints
    A = r.A' * r.A + r.c * r.c'
    B = - r.A'
    C = - r.c
    D = Diagonal(ones(size(s, 1)))
    E = zeros(size(s))
    F = 1

    # if !coneSatisfied(r, s, t) | (λCone > 0)
    #     ns = norm(s, 2)
    #
    #     D += s * s' / (ns^2)
    #     E += s / ns
    #     F += 1
    # end

    # if !coneSatisfied(r, s, t) | (λCone > 0)
    #     ns = norm(s, 2)
    #
    #     # D += s * s' / (ns^2)
    #     E += - s / ns
    #     F += 1
    # end

    # THIS SHOULD BE THE CORRECT ONE
    if !coneSatisfied(r, s, t) | (λCone > 0)
        ns = norm(s, 2)

        D += s * s' / (ns^2)
        E += - s / ns
        F += 1
    end

    # if !coneSatisfied(r, s, t) | (λCone > 0)
    #     ns = norm(s, 2)
    #
    #     D += Diagonal(sign.(s))
    #     E += - s / ns
    #     F += 1
    # end

    # if !coneSatisfied(r, s, t) | (λCone > 0)
    #     ns = norm(s, 2)
    #
    #     D += Diagonal(s)
    #     E += - s / ns
    #     F += 1
    # end

    hess = [A B C; B' D E; C' E' F]
    return Symmetric(hess)
end

runTestsAffine = false
runTestsOldCone = false
runTestsNewCone = false

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
end

if runTestsNewCone
    dslackCone1 = AL_coneSlack([-1 1], [0], [1; 1], 0)
    #AL_coneSlack([2 1/3; 1/3 3/4; 1 1], [3; 4; 5], [2; 1], -8)

    println("\n** Cone With Slack Variables")
    display(dslackCone1)

    xRan = -1:1
    s = [6] #[2; 5; -4]
    t = 5
    println("Probing: (with s = $s and t = $t): ")
    println([[x; x] for x in xRan])
    println("Raw Values: ")
    println([getRaw(dslackCone1, [x; x], s, t) for x in xRan])
    println("Satisfied: ")
    println([whichSatisfied(dslackCone1, [x; x], s, t) for x in xRan])
    println("Projections: ")
    println([getProjVecs(dslackCone1, [x; x], s, t) for x in xRan])
    println("Violations: ")
    println([getNormToProjVals(dslackCone1, [x; x], s, t) for x in xRan])

    println("\n\nGradient")
    xtest = [-5; 6]
    stest = [10] #dslackCone1.A * xtest
    ttest = 5 #dslackCone1.c' * xtest
    println("Probing: x = $xtest, s = $stest, t = $ttest")
    print("Which satisfied: ")
    println(whichSatisfied(dslackCone1, xtest, stest, ttest))
    jc = getGradC(dslackCone1, xtest, stest, ttest, true)
    display(reshape(jc, size(jc)))
    print("Violation: ")
    cCurr = getNormToProjVals(dslackCone1, xtest, stest, ttest)
    println(cCurr)
    println("Overall: -j'c")
    println(-jc'*cCurr)

    println("\n\nHessian")
    hc = getHessC(dslackCone1, xtest, stest, ttest)
    display(hc)

    if true
        dslackCone2 = AL_coneSlack([2 1/3; 1/3 3/4; 1 1], [3; 4; 5], [2; 1], -8)
        xtest = [-5; 6]
        stest = dslackCone2.A * xtest
        ttest = dslackCone2.c' * xtest
        println("\n\nLarge Hessian")
        hc = getHessC(dslackCone2, xtest, stest, ttest)
        display(hc)
    end

end
