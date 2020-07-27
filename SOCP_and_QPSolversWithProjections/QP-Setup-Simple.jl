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


# --------------------------
# Lagrangian
# φ(x) = f(x) + (ρ/2) ||c(x)||_2^2 + λ c(x)
#      = f(x) + (ρ/2) c(x)'c(x)    + λ c(x)
# --------------------------
# lambdaInit = zeros(size(fObj(x0)))
# rhoInit = 1
# phi(x) = fObj(x) + (rhoInit / 2) * cPlus(A, x, b)'cPlus(A, x, b) +
#                             lambdaInit' * cPlus(A, x, b)
#
# print("Example evaluation of the lagrangian at $x0: ")
# println(phi(x0))

mutable struct augLagQP_AffineIneq
    obj::objectiveQP
    constraints::AL_AffineInequality
    rho
    lambda
end

function evalAL(alQP::augLagQP_AffineIneq, x)
    # φ(x) = [(1/2) xT Q x + cT x] + (ρ/2) (Ax - b)'(Ax -b) + λ (Ax - b)
    fCurr = fObjQP(alQP.obj, x)
    cCurr = getNormToProjVals(alQP.constraints, x)

    return fCurr + (alQP.rho / 2) * cCurr'cCurr + (alQP.lambda)' * cCurr
end

function evalGradAL(alQP::augLagQP_AffineIneq, x)
    # ∇φ(x) = ∇f(x) + (ρ c(x) + λ) ∇c(x) becomes
    # [Qx + c] + [ρ (Ax - b) + λ] * A
    gradfCurr = dfdxQP(alQP.obj, x)           # This is Qx + c
    cCurr = getNormToProjVals(alQP.constraints, x)
    gradcCurr = getGradC(alQP.constraints, x) # The adjusted A matrix
    return gradfCurr + gradcCurr' * (alQP.rho * cCurr + alQP.lambda)
end

function evalHessAl(alQP::augLagQP_AffineIneq, x)
    # ∇^2φ(x) = ∇^2f(x) + ((ρ c(x) + λ) ∇^2c(x) + ρ ∇c(x) * ∇c(x)) becomes
    # Q + ρ A * A
    gradcCurr = getGradC(alQP.constraints, x) # The adjusted A matrix
    return alQP.obj.Q + alQP.rho * gradcCurr'gradcCurr
end


function getNormRes(resArr, floor = 10^(-20))
    # Take the norm to get non-negative.
    # Take max with floor to get positive.
    return max.(norm.(resArr), floor)
end

function calcNormGradResiduals(alQP::augLagQP_AffineIneq, xArr)
    #=
    Calculate the residuals where the AL is the merit function.
    Returns the norm of the gradient of the AL at each point in xArr
    =#
    resArr = [evalGradAL(alQP, x) for x in xArr]
    return getNormRes(resArr)
end

function calcALArr(alQP::augLagQP_AffineIneq, xArr)
    #=
    Calculate the value of the Augmented Lagrangian at each point
    =#
    return [evalAL(alQP, x) for x in xArr]
end
