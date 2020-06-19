

# QP Set-Up
# By definition, the QP has the following Properties
# minimize_x (1/2) xT Q x + cT x
# subject to Ax ⩽ B
#
# where (all are Real)
# Q is an nxn Symmetric Matrix (that is positive definite for convex)
# x is an nx1 vector
# c is an nx1 vector
# A is an mxn Matrix
# b is an mx1 vector

using LinearAlgebra

function checkPosDef(Q)
    # To check Positive Definite, we check that the eigenvals are positive
    # and that the determinant is positive.
    eVals = eigvals(Q)
    numRows = size(Q, 1)
    return (eVals > vec(zeros(numRows, 1))) && (det(Q) > 0)
end

# ---------------------------
# Objective Function
# (1/2) xT Q x + cT x
# Q is an nxn Symmetric Matrix (that is positive definite for convex
# c is an nx1 vector
# --------------------------

QMat = Symmetric([6 5; 0 8])
print("Q Matrix is Positive Definite: ")
println(checkPosDef(QMat))

cVec = [4; -3]

# Input x as a COLUMN vector (i.e. x = [4; 3])
fObj(x) = (1/2) * x'QMat*x + cVec'x

xExample = [5; 3]
print("Example evaluation of the objective function at $xExample: ")
println(fObj(xExample))


# ---------------------------
# Constraint Function
# Ax ≦ b
# A is an mxn Matrix
# b is an mx1 vector
# --------------------------

AMat = []
