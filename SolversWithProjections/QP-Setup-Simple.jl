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

include("feasibleCheck.jl")


function checkPosDef(Q)
    # To check Positive Definite, we check that the eigenvals are positive
    # and that the determinant is positive.
    eVals = eigvals(Q)
    numRows = size(Q, 1)
    return (eVals > vec(zeros(numRows, 1))) && (det(Q) > 0)
end

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

# Option 1: Symmetric([6 5; 0 8])
# Option 2: Symmetric([5 -0.5; 0 10])
QMat = Symmetric([6 5; 0 8])
println("\n")
print("Q Matrix is Positive Definite: ")
println(checkPosDef(QMat))

# With Option 1
# For an Interior Point, try [4; -3]
# For an Exterior Point, try [-30; 70]
# With Option 2
# For an Exterior Point, try [12, -70]
cVec = [4; -3]

# Input x as a COLUMN vector (i.e. x = [4; 3])
fObj(x) = (1/2) * x'QMat*x + cVec'x
dfObjdx(x) = QMat*x + cVec

print("Example evaluation of the objective function at $x0: ")
println(fObj(x0))


# ---------------------------
# Constraint Function
# Ax ≤ b
# A is an mxn Matrix
# b is an mx1 vector
# --------------------------

# 4x + 5y ≤ 20 and -4x + 5y ≤ 30 and y > -1
AMat = [4 5; -4 5; 0 -1]
bVec = [20; 30; 1]

# Check if the initial point is feasible
print("Is the initial point feasible? ")
println(isFeasiblePolyHedron(AMat, bVec, x0))

# --------------------------
# Lagrangian
# --------------------------
