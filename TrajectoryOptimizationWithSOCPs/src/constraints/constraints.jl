
using SparseArrays
abstract type constraint end

# Constraint Types and settings

include("..\\other_utils\\feasibleCheck.jl")
include("projections.jl")

#= Augmented Lagrangian Constraints
These are presented in the order:
- AffineEquality
- AffineInequality
- (Old) SOCP Constraints
- Functional SOCP Constraints with Slack Variables
- Simple SOCP Constraints
=#

# ---------------------
# Equality constraints
# ---------------------
"""
    AL_AffineEquality(A, b)

Creates an Affine Equality Constraint of the form `Ax = b`.

To check constraint satisfaction, use:
[`satisified`](@ref) and [`whichSatisfied`](@ref)

To evaluate the constraint, use:
- [`getRaw`](@ref) to evaluate `Ax - b`
- [`getNormToProjVals`](@ref) to evaluate the projection to `Ax = b`
"""
struct AL_AffineEquality <: constraint
    A
    b
end

# Check that a constraint is satisfied (Ax - b = 0)
"""
    satisfied(r::AL_AffineEquality, x) = (r.A * x == r.b)

Check constraint satisfaction. See also [`whichSatisfied`](@ref)
"""
satisfied(r::AL_AffineEquality, x) = (r.A * x == r.b)

"""
    whichSatisfied(r::AL_AffineEquality, x) = (r.A * x .== r.b)

Check constraint satisfaction. See also [`satisfied`](@ref)
"""
whichSatisfied(r::AL_AffineEquality, x) = (r.A * x .== r.b)

# Evaluate the constraint.
"""
    getRaw(r::AL_AffineEquality, x) = r.A * x - r.b

Evaluate the constraint without projection
"""
getRaw(r::AL_AffineEquality, x) = r.A * x - r.b

"""
    getProjVecs(r::AL_AffineEquality, x)

For every row, get the projection onto the line `a'x = b`.

See [`projAffineEq`](@ref) for more information.
"""
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

"""
    getNormToProjVals(r::AL_AffineEquality, x)

Get the row by row constraint violation based on projections. This is better
than [`getRaw`](@ref) since it doesn't depend on the magnitude of the elements
in `r.A` and `r.b`.

See also [`AL_AffineEquality`](@ref), [`getProjVecs`](@ref),
[`projAffineEq`](@ref)
"""
function getNormToProjVals(r::AL_AffineEquality, x)
    #=
    We want to get the projected vector and calculate the distance between the
    original point and the constraint.
    =#
    projVecs = getProjVecs(r, x)
    pvsDiff = [pv - x for pv in projVecs]

    # NaN removal due to floating point error near 0.0
    return replace(norm.(pvsDiff, 2), NaN=>0.0)
end

# Lastly, we do some calculus
"""
    getGradC(r::AL_AffineEquality, x)

Calculates the gradient of a constraint.

For `r::AL_AffineEquality`, `∇c(x) = A`

The variable "x" is passed to the header to maintain the same function header
but it is not used in the function.
"""
function getGradC(r::AL_AffineEquality, x)
    #=
    We want the gradient of the constraints. This is piecewise in the
    inequality case. But it is a single function in the equality case
    =#
    return r.A
end

"""
    getHessC(r::AL_AffineEquality)

Calculates the hessian of a constraint.

For `r::AL_AffineEquality`, `H(c(x)) = 0`
"""
function getHessC(r::AL_AffineEquality)
    #=
    We want the hessian of the constraints. This is just zero for affine
    constraints, but we need to match the dimensions.
    =#
    return spzeros(size(r.A, 2), size(r.A, 2)) # nxn
end

"""
    getHessC_ALTerm(r::AL_AffineEquality, x, rho = 1)

Calculates the hessian of the augmented lagrangian constraint term.

For `r::AL_AffineEquality`, `H(ρ c(x)'c(x) + λ c(x)) = ρ A'A`

The variable "x" is passed to the header to maintain the same function header
but it is not used in the function.
"""
function getHessC_ALTerm(r::AL_AffineEquality, x, rho = 1)
    #=
    The augmented lagrangian constraint term is of the form:
    ρ c(x)'c(x) + λ c(x)

    where c(x) = Ax - b.

    The Hessian matrix is then
    Htot = ρ (c.H + J.J + H.c) + λ H

    But H = 0 for Affine equalities, so
    Htot = ρ (J.J) = ρ A'A
    =#

    jacob = getGradC(r, x)
    return rho * jacob'jacob
end


# -----------------------
# Inequality constraints
# -----------------------

"""
    AL_AffineInequality(A, b)

Creates an Affine Inequality Constraint of the form `Ax ≤ b`.

To check constraint satisfaction, use:
[`satisified`](@ref) and [`whichSatisfied`](@ref)

To evaluate the constraint, use:
- [`getRaw`](@ref) to evaluate `Ax - b`
- [`getNormToProjVals`](@ref) to evaluate the projection to `Ax ≤ b`
"""
struct AL_AffineInequality <: constraint
    A
    b
end

# Check that a constraint is satisfied (Ax - b ≤ 0)
"""
    satisfied(r::AL_AffineInequality, x)

Check constraint satisfaction. See also [`whichSatisfied`](@ref)
"""
satisfied(r::AL_AffineInequality, x) = isFeasiblePolyHedron(r.A, r.b, x)

"""
    whichSatisfied(r::AL_AffineInequality, x)

Check constraint satisfaction. See also [`satisfied`](@ref)
"""
whichSatisfied(r::AL_AffineInequality, x) = (r.A * x .≤ r.b)

# Evaluate the constraint.
"""
    getRaw(r::AL_AffineInequality, x) = r.A * x - r.b

Evaluate the constraint without projection
"""
getRaw(r::AL_AffineInequality, x) = r.A * x - r.b

"""
    getProjVecs(r::AL_AffineInequality, x)

For every row, get the projection onto the line `a'x = b`.

See [`projAffineEq`](@ref) for more information.
"""
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

"""
    getNormToProjVals(r::AL_AffineInequality, x)

Get the row by row constraint violation based on projections. This is better
than [`getRaw`](@ref) since it doesn't depend on the magnitude of the elements
in `r.A` and `r.b`. It also automatically accounts for the inequality.

See also [`AL_AffineInequality`](@ref), [`getProjVecs`](@ref),
[`projAffineEq`](@ref)
"""
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
"""
    getGradC(r::AL_AffineInequality, x)

Calculates the gradient of a constraint.

For `r::AL_AffineInequality`, `∇c(x) = A` for rows where the constraint is NOT
satisfied.

Unlike the [`AL_AffineEquality`](@ref) case, the variable "x" is passed to
is needed to determine if the constraint is satisfied.
"""
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

"""
    getHessC(r::AL_AffineEquality)

Calculates the hessian of a constraint.

For `r::AL_AffineInequality`, `H(c(x)) = 0`
"""
function getHessC(r::AL_AffineInequality)
    #=
    We want the hessian of the constraints. This is just zero for affine
    constraints, but we need to match the dimensions.
    =#
    return spzeros(size(r.A, 2), size(r.A, 2)) # nxn
end

"""
    getHessC_ALTerm(r::AL_AffineInequality, x, rho = 1)

Calculates the hessian of the augmented lagrangian constraint term.

For `r::AL_AffineEquality`, `H(ρ c(x)'c(x) + λ c(x)) = ρ A+'A+`

The variable "x" is passed to the header to maintain the same function header
but it is not used in the function.
"""
function getHessC_ALTerm(r::AL_AffineInequality, x, rho = 1)
    #=
    The augmented lagrangian constraint term is of the form:
    ρ c(x)'c(x) + λ c(x)

    where c(x) = Ax - b.

    The Hessian matrix is then
    Htot = ρ (c.H + J.J + H.c) + λ H

    But H = 0 for Affine equalities, so
    Htot = ρ (J.J) = ρ A'A
    =#

    jacob = getGradC(r, x)
    return rho * jacob'jacob
end



# -----------------------
# Second-Order Cone Constraints
# -----------------------
# Specifically has the form ||Ax - b||_p ≤ c'x - d
# Where p denotes which norm (usually p = 2)

"""
Don't use this version. It is not yet functional. See [`AL_coneSlack`](@ref)
instead
"""
struct AL_pCone <: constraint
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
@doc raw"""
    AL_coneSlack(A, b, c, d)

Creates a Second Order Cone Constraint (SOCP) that uses slack variables.

We start with
```math
||Ax - b|| ≤ c^{\top} x - d
```

Using Slack variables, this becomes:
```math
||s|| ≤ t
```
```math
Ax - b = s
```
```math
c^{\top} x - d = t
```

So, in more standard form, this is becomes:
```math
||s|| - t ≤ 0
```
```math
Ax - b - s = 0
```
```math
c^{\top}x - d - t = 0
```

To check constraint satisfaction, use:
[`satisified`](@ref), [`whichSatisfied`](@ref), and [`coneSatisfied`](ref)

To evaluate the constraint, use:
- [`getRaw`](@ref) to evaluate each of `||s|| - t`, etc.
- [`getNormToProjVals`](@ref) to evaluate the projection to the affine
equalities and (more critically) the second order cone.

If the original `||Ax - b|| ≤ c'x - d` violation is needed, use
[`coneValOriginal`](@ref)

---

This ONLY works for second order cones. It will NOT work for any other p-cone,
i.e.
```math
||s||_2
```
works, but
```math
||s||_{10}
```
is not covered under this framework.
"""
struct AL_coneSlack <: constraint
    A
    b
    c
    d
end

# Evaluate the constraint.
"""
    getRaw(r::AL_coneSlack, x, s, t)

Evaluate the constraint without projection. See [`AL_coneSlack`](@ref)
"""
getRaw(r::AL_coneSlack, x, s, t) = [norm(s, 2) - t;
                                    r.A * x - r.b - s;
                                    r.c'x - r.d - t]

"""
    coneSatisfied(r::AL_coneSlack, s, t)

Checks if `||s|| - t ≤ 0`. See [`AL_coneSlack`](@ref)
"""
function coneSatisfied(r::AL_coneSlack, s, t)
    return (norm(s, 2) - t ≤ 0)
end

@doc raw"""
    coneValOriginal(r::AL_coneSlack, x)

Evaluates the raw cone value without slack variables, namely:
```math
||Ax - b|| ≤ c^{\top} x - d
```

See [`AL_coneSlack`](@ref)
"""
function coneValOriginal(r::AL_coneSlack, x)
    sInt = norm(r.A * x - r.b, 2)
    tInt = r.c'x - r.d
    return sInt - tInt
end

"""
    whichSatisfied(r::AL_coneSlack, x, s, t)

Check constraint satisfaction. See also [`satisfied`](@ref)
"""
function whichSatisfied(r::AL_coneSlack, x, s, t)
    raw = getRaw(r, x, s, t)
    return [raw[1] ≤ 0; raw[2:end] .== 0]
end

"""
    satisfied(r::AL_coneSlack, x, s, t)

Check constraint satisfaction. See also [`whichSatisfied`](@ref)
"""
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

"""
    coneActive(s, t, λ)

Checks if the cone constraint is "active." This prevents the solver from
getting stuck due to an inability to accurately update the dual λ.

We want to "activate" this constraint when `||s|| - t > 0` OR `λ > 0`.
"""
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
            # This is the "Solver might be stuck case."
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

"""
    getProjVecs(r::AL_coneSlack, x, s, t, filled = true, verbose = false)

Gets the projection vectors for each of the constraints that make up
[`AL_coneSlack`](@ref). In particular, we project onto the second order cone
and project each of the two affine equalities.

For the exact projections, see [`projSecondOrderCone`](@ref) and
[`projAffineEq`](@ref).

---

The parameter "filled" adjusts the projections onto the solid second order
cone (`||s|| ≤ t`) if `true` or the boundary of the second order cone
(`||s|| = t`) if `false`.

This is toggled based on [`coneActive`](@ref)
"""
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

"""
    getNormToProjVals(r::AL_coneSlack, x, s, t, λ = 0)

Get the row by row constraint violation based on projections for the affine
constraints. This is better than [`getRaw`](@ref) since it uses the minimum
distance to the second-order cone rather than the 1D evaluation of `||s|| - t`.
It is also better than [`getRaw`](@ref) since it doesn't depend on
the magnitude of the elements in `r.A` and `r.b`.

See also [`AL_coneSlack`](@ref), [`getProjVecs`](@ref),
[`projSecondOrderCone`](@ref), [`projAffineEq`](@ref)

---

The parameter `λ` describes the dual of `||s|| - t` and is used to determine
if the constraint is active. See [`coneActive`](@ref) for more information.
"""
function getNormToProjVals(r::AL_coneSlack, x, s, t, λ = 0)
    #=
    We want to get the projected vector and calculate the distance between the
    original point and the constraint.
    =#
    active, filled = coneActive(s, t, λ)
    projVec, signs = getProjVecs(r, x, s, t, filled)#true)#filled)
    coneDiff = norm([s; t] - projVec[1], 2)
    if !filled#false#!filled
        coneDiff *= -1
    end
    projDiff = [sign * norm(pv - x, 2) for (sign, pv) in
                                            zip(signs, projVec[2:end])]
    return [coneDiff; projDiff]
end

# Lastly, we do some calculus
"""
    getGradC(r::AL_coneSlack, x, s, t, verbose = false)

Calculates the gradient of a constraint.

For `r::AL_coneSlack`, this is actually the Jacobian with respect to the
variables `x`, `s`, and `t` in that order.
"""
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

"""
    getHessC(r::AL_coneSlack, x, s, t, λCone = 0)

Calculates the hessian of a constraint.

For `r::AL_coneSlack`, this is the hessian of the *constraint term* as it
appears in the augmented lagrangian!! Namely, `H(ρc(x)'c(x) + λ c(x))`

returns H, but user must multiply by ρ.

See the source code for more technical details.
"""
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
