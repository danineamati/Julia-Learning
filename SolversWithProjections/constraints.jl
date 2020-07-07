

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
    return zeros(size(r.A, 2), size(r.A, 2))
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
    return zeros(size(r.A, 2), size(r.A, 2))
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
# function getGradC(r::AL_pCone, x)
#     #=
#     We want the gradient of the constraints. This is piecewise in the
#     inequality case. But it is a single function in the equality case
#     =#
#     APost = zeros(size(r.A))
#     rowPassed = whichSatisfied(r, x)
#     for row in 1:size(r.A, 1)
#         if !rowPassed[row]
#             # Constraint is not met
#             APost[row, :] = r.A[row, :]
#         end
#     end
#     return APost
# end
#
# function getHessC(r::AL_pCone)
#     #=
#     We want the hessian of the constraints. This is just zero for affine
#     constraints, but we need to match the dimensions.
#     =#
#     return zeros(size(r.A, 2), size(r.A, 2))
# end


runTests = true

if runTests
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

    println("\nSecond-Order Constraints")
    dconeT = AL_pCone([5][:,:], vcat([-4]), vcat([3]), -8, 2)
    xRan = -5:5
    println("** Simplest")
    println("xVals = $(collect(xRan))")
    println("Raw Vals = $([getRaw(dconeT, vcat([x])) for x in xRan])")
    println("Satisfied = $([satisfied(dconeT, vcat([x])) for x in xRan])")
    println("Projection = $([getProjVecs(dconeT, vcat([x])) for x in xRan])")
    println("Violation = $([getNormToProjVals(dconeT, vcat([x])) for x in xRan])")
    dconeT2D = AL_pCone([5 3; 3 4], [-4; -2], [3; 4], -8, 2)
    xRan = -4:5
    println("** 2D")
    println("xVals = $(collect(xRan))")
    println("In the form [x; x]")
    println("Raw Vals = $([getRaw(dconeT2D, [x; x]) for x in xRan])")
    println("Satisfied = $([satisfied(dconeT2D, [x; x]) for x in xRan])")
    projVecList = [getProjVecs(dconeT2D, [x; x], true) for x in xRan]
    println("Projection = ")
    display(projVecList)
    println("Violation = $([getNormToProjVals(dconeT2D, [x; x]) for x in xRan])")
end
