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

include("constraints.jl")


# -------------------------
# Solver Parameters
# -------------------------
"""
Line Search Parameters (Notation from Toussaint Notes (2020))

`paramA::Float16`         # Used in Line Search, should be [0.01, 0.3]

`paramB::Float16`         # Used in Line Search, should be [0.1, 0.8]

----
Iteration Counter Parameters

`maxOuterIters::Int32`    # Number of Outer Loop iterations

`maxNewtonSteps::Int32`   # Number of Newton Steps per Outer Loop iterations

----
Exit Condition Parameter

`rTol::Float64`           # When steps are within rTol, loop will stop.

----
Penalty Update Parameters

`penaltyStep::Float16`    # Multiplies the penalty parameter per outer loop

`penaltyMax::Float64`     # Maximum value of the penalty parameter

----
Trust Region Parameters (Notation from Nocedal et Yuan (1998))

`trSizeStart::Float32`    # Starting size of the trust region

`trc1::Float16`           # Success Size Increase Parameter (1 < c1)

`trc2::Float16`           # Taylor Series Error Parameter (0 < c2 < 1)

`trc3::Float16`           # Failed Size Reduction Parameter (0 < c3 < c4)

`trc4::Float16`           # Failed Size Reduction Parameter (c3 < c4 < 1)
"""
struct solverParams
    # Line Search Parameters (Notation from Toussaint Notes (2020))
    paramA::Float16         # Error allowed should be [0.01, 0.3]
    paramB::Float16         # Reduction Factor should be [0.1, 0.8]
    # Iteration Counter Parameters
    maxOuterIters::Int32    # Number of Outer Loop iterations
    maxNewtonSteps::Int32   # Number of Newton Steps per Outer Loop iterations
    # Exit Condition Parameter
    rTol::Float64           # When steps are within rTol, loop will stop.
    # Penalty Update Parameters
    penaltyStep::Float16    # Multiplies the penalty parameter per outer loop
    penaltyMax::Float64     # Maximum value of the penalty parameter
    # Trust Region Parameters (Notation from Nocedal et Yuan (1998))
    trSizeStart::Float32    # Starting size of the trust region
    trc1::Float16           # Success Size Increase Parameter (1 < c1)
    trc2::Float16           # Taylor Series Error Parameter (0 < c2 < 1)
    trc3::Float16           # Failed Size Reduction Parameter (0 < c3 < c4)
    trc4::Float16           # Failed Size Reduction Parameter (c3 < c4 < 1)
end

"""
Prints the struct variables in the `solverParams` struct.

See also: [`solverParams`](@ref) for more information.
"""
function solParamPrint(sp::solverParams)
    println()
    println("Beginning solver with parameters: ")
    println("(Line Search) : a = $(sp.paramA), b = $(sp.paramB)")
    print("(Loop #)      : Outer = $(sp.maxOuterIters), ")
    println("Newton = $(sp.maxNewtonSteps)")
    println("(or End at)   : Δ(∇L) = $(sp.rTol)")
    println("(Penalty)     : Δρ = $(sp.penaltyStep), ρMax = $(sp.penaltyMax)")
    println("(Trust Region): Δ = $(sp.trSizeStart)")
    println("                c1 = $(sp.trc1) (with 1 < c1 -> $(1 < sp.trc1))")
    print("                c2 = $(sp.trc2) (with 0 < c2 < 1 -> ")
    println("$(0 < sp.trc2) & $(sp.trc2 < 1))")
    println("                c3 = $(sp.trc3), c4 = $(sp.trc4) ")
    print("                (with 0 < c3 < c4 < 1) -> $(0 < sp.trc3) & ")
    println("$(sp.trc3 < sp.trc4) & $(sp.trc4 < 1)")
end



"""
    SOCP_primals(x, s, t)

Struct to store the primal variables of the form [x; s; t]

`x` and `s` are mx1 and nx1 vectors.
`t` is a real number

See also: [`primalVec`](@ref) and [`primalStruct`](@ref)
"""
mutable struct SOCP_primals
    x
    s
    t
end

"""
    primalVec(y::SOCP_primals)

Converts [`SOCP_primals`](@ref) struct to [x; s; t] vector.
"""
function primalVec(y::SOCP_primals)
    return [y.x; y.s; y.t]
end

"""
    primalStruct(v, xSize::Int64, sSize::Int64, tSize::Int64)

Converts [x; s; t] vector to [`SOCP_primals`](@ref) struct
"""
function primalStruct(v, xSize::Int64, sSize::Int64, tSize::Int64)
    return SOCP_primals(v[1:xSize], v[xSize+1:xSize+sSize], v[end])
end


"""
    getXVals(yList::Array{SOCP_primals, 1})

Returns the "x" values of and an array of [`SOCP_primals`](@ref).
Used primarily with plotting.
"""
function getXVals(yList::Array{SOCP_primals, 1})
    xList = []
    for xst in yList
        push!(xList, xst.x)
    end
    return xList
end

@doc raw"""
    getViolation(yList::Array{SOCP_primals, 1}, c::AL_coneSlack)

returns how much each SOCP constraint is violated
(in 3 mulitdimensional arrays).

Recall that an SOCP is generally
```math
||Ax-b|| ≤ c^⊤ x - d
```

Using slack variables, we have

```math
||s|| - t ≤ 0
```

```math
(Ax - b) - s = 0
```

```math
(c^⊤ x - d) - t = 0
```

This function returns the constraint violation for each of the above 3 in 3
separate arrays.
"""
function getViolation(yList::Array{SOCP_primals, 1}, c::AL_coneSlack)
    coneViolation = []
    affineViolation = []
    lastViolation = []
    for y in yList
        vio = getNormToProjVals(c, y.x, y.s, y.t)
        push!(coneViolation, vio[1])
        push!(affineViolation, vio[2:1+size(y.s, 1)])
        push!(lastViolation, vio[end])
    end
    return coneViolation, affineViolation, lastViolation
end

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
    augLagQP_2Cone(obj::objectiveQP, constraints::AL_coneSlack, ρ, λ)

This is a mutable struct that stores an SOCP Augmented Lagrangian.

For a single SOCP constraint, the lagrangian is
```math
φ(y) = f(x) + (ρ/2) ||c(y)||_2^2 + λ c(y)
```
```math
φ(y) = f(x) + (ρ/2) c(y)'c(y)    + λ c(y)
```

Where `y = [x; s; t]` is represented as an [`SOCP_primals`] struct.

Generally, we initialize `λ` to 0 and `ρ` to 1.

See also: [`evalAL`](@ref), [`evalGradAL`](@ref), [`evalHessAl`](@ref)
"""
mutable struct augLagQP_2Cone
    obj::objectiveQP
    constraints::AL_coneSlack # Currently the "p-cone" is only a "2-cone"
    rho
    lambda
end

"""
    evalAL(alQP::augLagQP_2Cone, y::SOCP_primals)

Evaluates the Augmented Lagrangian of an SOCP with the primals `y`

returns a real number
"""
function evalAL(alQP::augLagQP_2Cone, y::SOCP_primals)
    # φ(y) = f(x) + (ρ/2) c(y)'c(y)    + λ c(y)
    # φ(y) = [(1/2) xT Q x + cT x] + (ρ/2) (c(y))'(c(y)) + λ (c(y))
    fCurr = fObjQP(alQP.obj, y.x)
    cCurr = getNormToProjVals(alQP.constraints, y.x, y.s, y.t, alQP.lambda[1])

    return fCurr + (alQP.rho / 2) * cCurr'cCurr + (alQP.lambda)' * cCurr
end

"""
    evalGradAL(alQP::augLagQP_2Cone, y::SOCP_primals, verbose = false)

Evaluates the gradient of the Augmented Lagrangian of an SOCP with the
primals `y`

returns a vector of size `y`
"""
function evalGradAL(alQP::augLagQP_2Cone, y::SOCP_primals, verbose = false)
    # ∇φ(y) = ∇f(x) + J(c(y))'(ρ c(y) + λ)
    # y = [x; s; t]
    # ∇φ(y) is (n+m+1)x1
    gradfCurr = dfdxQP(alQP.obj, y.x)           # This is just Qx + c

    if verbose
        println("Size ∇xf(x) = $(size(gradfCurr))")
    end

    # Pad the gradient
    paddedGradf = [gradfCurr; zeros(size(y.s, 1) + size(y.t, 1))]

    if verbose
        println("Size ∇yf(x) = $(size(paddedGradf))")
    end

    cCurr = getNormToProjVals(alQP.constraints, y.x, y.s, y.t, alQP.lambda[1])
    # The Jacobian matrix
    jacobcCurr = getGradC(alQP.constraints, y.x, y.s, y.t)

    if verbose
        println("Size c(y) = $(size(cCurr))")
        println("Size λ = $(size(alQP.lambda))")
        println("Size J(c(y)) = $(size(jacobcCurr))")
    end

    cTotal = jacobcCurr' * (alQP.rho * cCurr + alQP.lambda)

    if verbose
        println("Size cTotal = $(size(cTotal))")
    end

    if false
        print("∇f = ")
        println(paddedGradf)
        print("∇c = ")
        print(cTotal)
        print(", with c = ")
        println(cCurr)
    end

    return paddedGradf + cTotal
end

"""
    evalHessAl(alQP::augLagQP_2Cone, y::SOCP_primals, verbose = false)

Evaluates the Hessian of the Augmented Lagrangian of an SOCP with the
primals `y`

returns a Symmetric Matrix of size `y`x`y`
"""
function evalHessAl(alQP::augLagQP_2Cone, y::SOCP_primals, verbose = false)
    #=
    H(φ(x)) = H(f(x)) + ((ρ c(x) + λ) H(c(x)) + ρ ∇c(x) * ∇c(x))
    We can group the last two terms into one. This yields
    H(φ(x))
    =#
    hSize = size(y.x, 1) + size(y.s, 1) + size(y.t, 1)
    xSize = size(y.x, 1)

    hessf1 = [alQP.obj.Q; zeros(hSize - xSize, xSize)]
    hessf2 = zeros(hSize, hSize - xSize)
    hessfPadded = [hessf1 hessf2]

    if verbose
        println("Size H(f) = $(size(hessfPadded))")
    end

    hesscCurr = getHessC(alQP.constraints, y.x, y.s, y.t)

    if verbose
        println("Size H(cT) = $(size(hesscCurr))")
    end

    return hessfPadded + alQP.rho * hesscCurr
end

"""
    getNormRes(resArr, floor = 10^(-20))

Acts as a safe 2-norm that operates on an array that would approach zero.
Usually this is used with semilog or log-log plots of the residuals

See also: [`safeNorm`](@ref), [`calcNormGradResiduals`](@ref)
"""
function getNormRes(resArr, floor = 10^(-20))
    # Take the norm to get non-negative.
    # Take max with floor to get positive.
    return max.(norm.(resArr), floor)
end

"""
    calcNormGradResiduals(alQP::augLagQP_2Cone,
                          yList::Array{SOCP_primals, 1})

Calculates the 2-norm of the residuals of an augmented lagrangian at each of
the inputs `y`. The 2-norm in safe in that it is bounded below (e.g. by
10^(-20)) and can be plotted on a semilog or log-log plot.

See also: [`getNormRes`](@ref)
"""
function calcNormGradResiduals(alQP::augLagQP_2Cone,
                               yList::Array{SOCP_primals, 1})
    #=
    Calculate the residuals where the AL is the merit function.
    Returns the norm of the gradient of the AL at each point in xArr
    =#
    resArr = [evalGradAL(alQP, y) for y in yList]
    return getNormRes(resArr)
end

"""
    calcALArr(alQP::augLagQP_2Cone, yList::Array{SOCP_primals, 1})

Evaluates the augmented lagrangian at each input `y`.
"""
function calcALArr(alQP::augLagQP_2Cone, yList::Array{SOCP_primals, 1})
    #=
    Calculate the value of the Augmented Lagrangian at each point
    =#
    return [evalAL(alQP, y) for y in yList]
end
