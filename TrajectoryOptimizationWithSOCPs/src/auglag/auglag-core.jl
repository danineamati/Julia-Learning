# We have a similar QP set-up as before

# Simply include this file

# QP Set-Up
# By definition, the QP has the following Properties
# minimize_x (1/2) xT Q x + cT x
# subject to Ax ≤ B
#
# where (all are Real)
# Q is an nxn Symmetric Matrix (that is positive definite for convex)
# x is an nx1 vector
# c is an nx1 vector
# A is an mxn Matrix
# b is an mx1 vector

using LinearAlgebra

include("..\\objective\\QP_Linear_objectives.jl")
include("..\\constraints\\constraintManager.jl")
include("..\\other_utils\\utils.jl")


# --------------------------
# Lagrangian
# φ(y) = f(x) + (ρ/2) ||c(y)||_2^2 + λ c(y)
#      = f(x) + (ρ/2) c(y)'c(y)    + λ c(y)
# --------------------------
# Where y = [x; s; t]
# --------------------------
# lambdaInit = zeros(size(fObj(x0)))
# rhoInit = 1

@doc raw"""
    augLag(obj::objectiveFunc, cM::constraintManager, rho::Float64)

This is a mutable struct that stores an SOCP Augmented Lagrangian.

For a single SOCP constraint, the lagrangian is
```math
φ(y) = f(y) + (ρ/2) ||c(y)||_2^2 + λ c(y)
```
```math
φ(y) = f(y) + (ρ/2) c(y)'c(y)    + λ c(y)
```

Where `y` is represented a primal vector struct.

Generally, we initialize `ρ` to 1.

See also: [`evalAL`](@ref), [`evalGradAL`](@ref), [`evalHessAl`](@ref)
"""
mutable struct augLag
    obj::objectiveFunc
    cM::constraintManager
    rho::Float64
end

"""
    evalAL(alQP::augLag, y)

Evaluates the Augmented Lagrangian with the primals `y`

returns a real number
"""
function evalAL(alQP::augLag, y)
    # φ(y) = f(x) + (ρ/2) c(y)'c(y)    + λ c(y)
    # φ(y) = [(1/2) xT Q x + cT x] + (ρ/2) (c(y))'(c(y)) + λ (c(y))
    fCurr = fObjQP(alQP.obj, y)[1]
    cCurr = evalConstraints(alQP.cM, y, alQP.rho)
    # println("f(x) = $fCurr")
    # println(" with size -> $(size(fCurr))")
    # println("c(x) = $cCurr")
    # println(" with size -> $(size(cCurr))")
    return fCurr + cCurr
end

"""
    evalGradAL(alQP::augLag, y::SOCP_primals, verbose = false)

Evaluates the gradient of the Augmented Lagrangian with the primals `y`.

returns a vector of size `y`
"""
function evalGradAL(alQP::augLag, y)
    # ∇φ(y) = ∇f(x) + J(c(y))'(ρ c(y) + λ)
    # y = [x; s; t]
    # ∇φ(y) is (n+m+1)x1
    gradfCurr = dfdxQP(alQP.obj, y)
    gradCCurr = evalGradConstraints(alQP.cM, y, alQP.rho)

    return gradfCurr + gradCCurr
end

"""
    evalHessAl(alQP::augLag, y::SOCP_primals, verbose = false)

Evaluates the Hessian of the Augmented Lagrangian of an SOCP with the
primals `y`

returns a Symmetric Matrix of size `y`x`y`
"""
function evalHessAl(alQP::augLag, y, verbose = false)

    hessf = hessQP(alQP.obj)
    hessC = evalHessConstraints(alQP.cM, y)
    return hessf + alQP.rho * hessC
end

"""
    calcNormGradResiduals(alQP::augLag,
                          yList::Array{SOCP_primals, 1})

Calculates the 2-norm of the residuals of an augmented lagrangian at each of
the inputs `y`. The 2-norm in safe in that it is bounded below (e.g. by
10^(-20)) and can be plotted on a semilog or log-log plot.

See also: [`safeNorm`](@ref)
"""
function calcNormGradResiduals(alQP::augLag, yList)
    #=
    Calculate the residuals where the AL is the merit function.
    Returns the norm of the gradient of the AL at each point in xArr
    =#
    resArr = [evalGradAL(alQP, y) for y in yList]
    return safeNorm(resArr)
end

"""
    calcALArr(alQP::augLag, yList::Array{SOCP_primals, 1})

Evaluates the augmented lagrangian at each input `y`.
"""
function calcALArr(alQP::augLag, yList)
    #=
    Calculate the value of the Augmented Lagrangian at each point
    =#
    return [evalAL(alQP, y) for y in yList]
end
