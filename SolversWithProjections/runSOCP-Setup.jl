# We have a similar QP set-up as before, but now we declare instances
# or the key structs.

include("SOCP-Setup-Simple.jl")
include("constraints.jl")



# -------------------------
# Solver Parameters
# -------------------------
currSolveParams = solverParams(0.1, 0.5, 12, 3, 10^-10, 10, 10^6)
solParamPrint(currSolveParams)

# --------------------------
# Set an example initial starting point
# --------------------------
x0 = [-5; 0]

# ---------------------------
# Objective Function
# (1/2) xT Q x + cT x
# Q is an nxn Symmetric Matrix (that is positive definite for convex
# c is an nx1 vector
# --------------------------

# Examples used:
#           Q               |       c      |       x*        | name
# -----------------------------------------------------------------------
# Symmetric([6 5; 0 8])     |    [4; -3]   | [-47/23; 38/23] |  Interior
#                           |              | [-2.043; 1.652] |
# Symmetric([6 5; 0 8])     |   [-30; 70]  |   [35/6; -1]    |  Far Ext. (Edge)
# Symmetric([5 -0.5; 0 10]) |   [12, -70]  |   [-5/4; 5]     |  Up  (Vertex)

QPName = "Interior"

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
cVec = [4; -3]

# Generate the Struct
thisQP = objectiveQP(QMat, cVec)

print("Example evaluation of the objective function at $x0: ")
println(fObjQP(thisQP, x0))

# ---------------------------
# Constraint Function
# ||Ax - b|| - (c'x - d) ≤ 0
#
# Which we write as
# ||s|| ≤ t
# Ax - b = s
# c'x - d = t
#
# A is an mxn Matrix
# b is an mx1 vector
# c is an nx1 vector
# d is a real number
# x is nx1
# s is mx1
# t is a real number
# Note that a 2-norm is assumed
# --------------------------

AMat = [4 5; -4 5; 0 -1]
bVec = [20; 30; 1]
cVec = [2; -2]
dVal = -8

# Generate the struct
thisConstr = AL_coneSlack(AMat, bVec, cVec, dVal)

# Now initialize s and t
s0 = AMat * x0 - bVec
t0 = cVec'x0 - dVal

y0 = SOCP_primals(x0, s0, t0)

# Check if the initial point is feasible
print("Is the initial point feasible? ")
println(satisfied(thisConstr, x0, s0, t0))


# --------------------------
# Lagrangian
# φ(y) = f(x) + (ρ/2) ||c(y)||_2^2 + λ c(y)
#      = f(x) + (ρ/2) c(y)'c(y)    + λ c(y)
# --------------------------
lambdaSize = size(bVec, 1) + 2

alcone = augLagQP_2Cone(thisQP, thisConstr, 1, zeros(lambdaSize))
print("Evaluating the Augmented Lagrangian at the starting value of $x0: ")
println(evalAL(alcone, y0))
println("Evaluating the AL gradient: $(evalGradAL(alcone, y0))")
println("Evaluating the AL hessian: ")
display(evalHessAl(alcone, y0))
