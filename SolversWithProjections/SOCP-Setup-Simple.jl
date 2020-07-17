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


function checkPosDef(Q)
    # To check Positive Definite, we check that the eigenvalues are positive
    # and that the determinant is positive.
    eVals = eigvals(Q)
    numRows = size(Q, 1)
    return (eVals > vec(zeros(numRows, 1))) && (det(Q) > 0)
end

# -------------------------
# Solver Parameters
# -------------------------

struct solverParams
    paramA::Float16         # Used in Line Search, should be [0.01, 0.3]
    paramB::Float16         # Used in Line Search, should be [0.1, 0.8]
    maxOuterIters::Int32    # Number of Outer Loop iterations
    maxNewtonSteps::Int32   # Number of Newton Steps per Outer Loop iterations
    xTol::Float64           # When steps are within xTol, loop will stop.
    penaltyStep::Float16    # Multiplies the penalty parameter per outer loop
    penaltyMax::Float64     # Maximum value of the penalty parameter
end

function solParamPrint(sp::solverParams)
    println()
    println("Beginning solver with parameters: ")
    println("(Line Search) : a = $(sp.paramA), b = $(sp.paramB)")
    print("(Loop #)      : Outer = $(sp.maxOuterIters), ")
    println("Newton = $(sp.maxNewtonSteps)")
    println("(or End at)   : Δx = $(sp.xTol)")
    println("(Penalty)     : Δρ = $(sp.penaltyStep), ρMax = $(sp.penaltyMax)")
end

# ---------------------------
# Objective Function
# (1/2) xT Q x + cT x
# Q is an nxn Symmetric Matrix (that is positive definite for convex
# c is an nx1 vector
# --------------------------

struct objectiveQP
    Q
    c
end

# Input x as a COLUMN vector (i.e. x = [4; 3])
function fObjQP(qp::objectiveQP, x)
    return (1/2) * x' * (qp.Q) * x + (qp.c)' * x
end

function dfdxQP(qp::objectiveQP, x)
    return (qp.Q) * x + qp.c
end

# ------------------------
# We have three primal variables: [x, s, t]
# We can stack them in one struct
# ------------------------

mutable struct SOCP_primals
    x
    s
    t
end

function primalVec(y::SOCP_primals)
    return [y.x; y.s; y.t]
end

function primalStruct(v, xSize::Int64, sSize::Int64, tSize::Int64)
    return SOCP_primals(v[1:xSize], v[xSize+1:xSize+sSize], v[end])
end

function getXVals(yList::Array{SOCP_primals, 1})
    xList = []
    for xst in yList
        push!(xList, xst.x)
    end
    return xList
end

function getViolation(yList::Array{SOCP_primals, 1}, c::AL_coneSlack)
    coneViolation = []
    affineViolation = []
    lastViolation = []
    for y in yList
        vio = getNormToProjVals(c, y.x, y.s, y.t)
        push!(coneViolation, vio[1])
        push!(affineViolation, vio[2:2+size(y.s, 1)])
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

mutable struct augLagQP_2Cone
    obj::objectiveQP
    constraints::AL_coneSlack # Currently the "p-cone" is only a "2-cone"
    rho
    lambda
end

function evalAL(alQP::augLagQP_2Cone, y::SOCP_primals)
    # φ(y) = f(x) + (ρ/2) c(y)'c(y)    + λ c(y)
    # φ(y) = [(1/2) xT Q x + cT x] + (ρ/2) (c(y))'(c(y)) + λ (c(y))
    fCurr = fObjQP(alQP.obj, y.x)
    cCurr = getNormToProjVals(alQP.constraints, y.x, y.s, y.t)

    return fCurr + (alQP.rho / 2) * cCurr'cCurr + (alQP.lambda)' * cCurr
end

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

    cCurr = getNormToProjVals(alQP.constraints, y.x, y.s, y.t)
    # The Jacobian matrix
    jacobcCurr = getGradC(alQP.constraints, y.x, y.s, y.t, true)

    if verbose
        println("Size c(y) = $(size(cCurr))")
        println("Size λ = $(size(alQP.lambda))")
        println("Size J(c(y)) = $(size(jacobcCurr))")
    end

    cTotal = jacobcCurr' * (alQP.rho * cCurr + alQP.lambda)

    if verbose
        println("Size cTotal = $(size(cTotal))")
    end

    print("∇f = ")
    println(paddedGradf)
    print("∇c = ")
    print(cTotal)
    print(", with c = ")
    println(cCurr)

    return paddedGradf + cTotal
end

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

    return hessfPadded + 2 * alQP.rho * hesscCurr
end


function getNormRes(resArr, floor = 10^(-20))
    # Take the norm to get non-negative.
    # Take max with floor to get positive.
    return max.(norm.(resArr), floor)
end

function calcNormGradResiduals(alQP::augLagQP_2Cone,
                                    yList::Array{SOCP_primals, 1})
    #=
    Calculate the residuals where the AL is the merit function.
    Returns the norm of the gradient of the AL at each point in xArr
    =#
    resArr = [evalGradAL(alQP, y) for y in yList]
    return getNormRes(resArr)
end

function calcALArr(alQP::augLagQP_2Cone, yList::Array{SOCP_primals, 1})
    #=
    Calculate the value of the Augmented Lagrangian at each point
    =#
    return [evalAL(alQP, y) for y in yList]
end
