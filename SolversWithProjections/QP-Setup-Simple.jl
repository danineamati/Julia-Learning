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

currSolveParams = solverParams(0.1, 0.5, 16, 2, 10^-10, 10, 10^6)
solParamPrint(currSolveParams)

# --------------------------
# Set an example initial starting point
# --------------------------

x0 = [-1; 0.5]

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

# Examples used:
#           Q               |       c      |       x*        | name
# -----------------------------------------------------------------------
# Symmetric([6 5; 0 8])     |    [4; -3]   | [-47/23; 38/23] |  Interior
# Symmetric([6 5; 0 8])     |   [-30; 70]  |   [35/6; -1]    |  Far Ext. (Edge)
# Symmetric([5 -0.5; 0 10]) |   [12, -70]  |   [-5/4; 5]     |  Up  (Vertex)


# Option 1: Symmetric([6 5; 0 8])
# Option 2: Symmetric([5 -0.5; 0 10])
QMat = Symmetric([6 5; 0 8])
println()
print("Q Matrix is Positive Definite: ")
println(checkPosDef(QMat))

# With Option 1
# For an Interior Point, try [4; -3]
# For an Exterior Point, try [-30; 70]
# With Option 2
# For an Exterior Point, try [12, -70]
cVec = [-30; 70]

# Generate the Struct
thisQP = objectiveQP(QMat, cVec)

# Input x as a COLUMN vector (i.e. x = [4; 3])
function fObjQP(qp::objectiveQP, x)
    return (1/2) * x' * (qp.Q) * x + (qp.c)' * x
end

function dfdxQP(qp::objectiveQP, x)
    return (qp.Q) * x + qp.c
end

print("Example evaluation of the objective function at $x0: ")
println(fObjQP(thisQP, x0))


# ---------------------------
# Constraint Function
# Ax ≤ b
# A is an mxn Matrix
# b is an mx1 vector
# --------------------------

# 4x + 5y ≤ 20 and -4x + 5y ≤ 30 and y > -1
AMat = [4 5; -4 5; 0 -1]
bVec = [20; 30; 1]

# Generate the struct
thisConstr = AL_AffineInequality(AMat, bVec)

# Check if the initial point is feasible
print("Is the initial point feasible? ")
println(satisfied(thisConstr, x0))

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
    fCurr = fObjQP(alQP.obj, x)
    cCurr = getNormToProjVals(alQP.constraints, x)

    return fCurr + (alQP.rho / 2) * cCurr'cCurr + (alQP.lambda)' * cCurr
end

function evalGradAL(alQP::augLagQP_AffineIneq, x)
    gradfCurr = dfdxQP(alQP.obj, x)           # This is Qx + c
    cCurr = getNormToProjVals(alQP.constraints, x)
    gradcCurr = getGradC(alQP.constraints, x) # The adjusted A matrix
    return gradfCurr + gradcCurr' * (alQP.rho * cCurr + alQP.lambda)
end

function evalHessAl(alQP::augLagQP_AffineIneq, x)
    gradcCurr = getGradC(alQP.constraints, x) # The adjusted A matrix
    return alQP.obj.Q + alQP.rho * gradcCurr'gradcCurr
end


alTest = augLagQP_AffineIneq(thisQP, thisConstr, 1, zeros(size(bVec)))
print("Evaluating the Augmented Lagrangian at the starting value of $x0: ")
println(evalAL(alTest, x0))
println("Evaluating the AL gradient: $(evalGradAL(alTest, x0))")
println("Evaluating the AL hessian: $(evalHessAl(alTest, x0))")
