

# Constraint Types and settings

include("feasibleCheck.jl")
include("projections.jl")

# Augmented Lagrangian Constraints

# ---------------------
# Equality constraints
# ---------------------
struct ALQP_AffineEquality
    A
    b
end

# Check that a constraint is satisfied (Ax - b = 0)
satisfied(r::ALQP_AffineEquality, x) = (r.A * x == r.b)
whichSatisfied(r::ALQP_AffineEquality, x) = (r.A * x .== r.b)

# Evaluate the constraint.
# Raw = Evaluate the function without projection
getRaw(r::ALQP_AffineEquality, x) = r.A * x - r.b
function getProjVals(r::ALQP_AffineEquality, x)
    # We want to project each constraint
    # We write Ax = b
    # But this is equivalent to a1'x = b1, a2'x = b2, ..., am'x = bm
    # Where ak is the row k in A
    projVec = []
    for ind in 1:size(r.b, 1)
        aI = r.A[ind, :]
        bI = r.b[ind]
        push!(projVec, projAffineEq(aI, bI, x))
    end

    return projVec
end

# -----------------------
# Inequality constraints
# -----------------------
struct ALQP_AffineInequality
    A
    b
end

# Check that a constraint is satisfied (Ax - b ≤ 0)
satisfied(r::ALQP_AffineInequality, x) = isFeasiblePolyHedron(r.A, r.b, x)
whichSatisfied(r::ALQP_AffineInequality, x) = (r.A * x .≤ r.b)




runTests = true

if runTests
    println()
    println("Affine Equality Constraints:")
    dT = ALQP_AffineEquality([5], [10])
    println("Simplest")
    println("On line  (True) : $(satisfied(dT, 2))")
    println("Off line (False): $(satisfied(dT, 4)) and $(satisfied(dT, 0))")
    dT2 = ALQP_AffineEquality([5 0; 0 6], [10; 12])
    println("2D")
    println("On Intesection (True)        : $(satisfied(dT2, [2; 2]))")
    println("Check which not (True, False): $(whichSatisfied(dT2, [2; 4]))")

    println("\nAffine Inequality Constraints")
    diT = ALQP_AffineInequality([5], [10])
    println("On line    (True) : $(satisfied(diT, 2))")
    println("Above line (False): $(satisfied(diT, 4))")
    println("Under line (True) : $(satisfied(diT, 0))")
    diT2 = ALQP_AffineInequality([5 0; 0 6], [10; 12])
    println("2D")
    println("On Intesection (True)        : $(satisfied(diT2, [2; 2]))")
    println("Check which not (True, False): $(whichSatisfied(diT2, [2; 4]))")
    println("Check which not (True, True) : $(whichSatisfied(diT2, [2; 0]))")

end
