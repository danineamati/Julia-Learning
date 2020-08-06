# Only NEW SOCP Constraints in comparison to the original SOCP solver

using SparseArrays

include("..\\other_utils\\feasibleCheck.jl")
include("projections.jl")


# -----------------------
# Simple Second-Order Cone Constraints
# -----------------------
@doc raw"""
    AL_coneSlack(tMax)

Creates a _simple_ Second Order Cone Constraint (SOCP) of the form
```math
||t|| ≤ tMax
```

So, in more standard form, this is becomes:
```math
||t|| - tMax ≤ 0
```

To check constraint satisfaction, use:
[`satisified`](@ref) and [`whichSatisfied`](@ref)

To evaluate the constraint, use:
- [`getRaw`](@ref) to evaluate `||t|| - tMax`.
- [`getNormToProjVals`](@ref) to evaluate the projection to the affine
equalities and (more critically) the second order cone.

"""
struct AL_simpleCone <: constraint
    tMax
end

"""
    getRaw(r::AL_simpleCone, t)

Evaluate the constraint without projection.
"""
getRaw(r::AL_simpleCone, t) = norm(t) - r.tMax

"""
    satisfied(r::AL_simpleCone, t)

Check constraint satisfaction. See also [`whichSatisfied`](@ref)
"""
satisfied(r::AL_simpleCone, t) = (getRaw(r, t) ≤ 0)

"""
    whichSatisfied(r::AL_simpleCone, tArr)

Check constraint satisfaction for each element in `tArr`.
"""
whichSatisfied(r::AL_simpleCone, tArr) = [satisfied(r, t) for t in tArr]

"""
    getProjVecs(r::AL_simpleCone, t, filled = true)

Gets the projection vectors for `t` onto the the second order cone defined
by `||t|| - tMax`.

See also [`AL_simpleCone`](@ref)
"""
function getProjVecs(r::AL_simpleCone, t, filled = true)
    return projSecondOrderCone(t, r.tMax, filled)
end

"""
    getNormToProjVals(r::AL_simpleCone, t, λ = 0)

Get the constraint violation for the second order cone constraints.

The parameter `λ` describes the dual of `||t|| - tMax` and is used to determine
if the constraint is active. See [`coneActive`](@ref) for more information.
"""
function getNormToProjVals(r::AL_simpleCone, t, λ = 0)

    active, filled = coneActive(t, r.tMax, λ)
    pv = getProjVecs(r, t, filled)

    return [norm([t; r.tMax] - pv)]
end

"""
    getGradC(r::AL_simpleCone, t)

Calculates the gradient of a constraint.

For `r::AL_simpleCone`, this is the `∇c(x) = x / ||x||`

There is a singularity at the tip.
"""
function getGradC(r::AL_simpleCone, t)
    return t / norm(t)
end

"""
    getHessC(r::AL_simpleCone, t)

Calculate the hessian of a constraint.

For `r::AL_simpleCone`, `H(c(x)) = 0`
"""
function getHessC(r::AL_simpleCone, t)
    return spzeros(size(t, 1), size(t, 1))
end

"""
    getHessC_ALTerm(r::AL_simpleCone, t, rho = 1)

For `r::AL_simpleCone`, `H(ρ c(x)'c(x) + λ c(x)) = ρ s * s' / norm(s)^2`
"""
function getHessC_ALTerm(r::AL_simpleCone, t, rho = 1)
    #=
    The augmented lagrangian constraint term is of the form:
    ρ c(x)'c(x) + λ c(x)

    where c(x) = x / ||x||.

    The Hessian matrix is then
    Htot = ρ (c.H + J.J + H.c) + λ H

    But H = 0 for Affine equalities, so
    Htot = ρ (J.J) = ρ s * s' / norm(s)^2
    =#

    return rho * (t * t') / (norm(t)^2)
end
