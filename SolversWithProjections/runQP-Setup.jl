# We have a similar QP set-up as before, but now we declare instances
# or the key structs.

include("QP-Setup-Simple.jl")
include("constraints.jl")



# -------------------------
# Solver Parameters
# -------------------------
currSolveParams = solverParams(0.1, 0.5, 10, 3, 10^-10, 10, 10^6)
solParamPrint(currSolveParams)

# --------------------------
# Set an example initial starting point
# --------------------------
x0 = [-5; 20]

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
# Symmetric([6 5; 0 8])     |   [-30; 70]  |   [35/6; -1]    |  Far Ext. (Edge)
# Symmetric([5 -0.5; 0 10]) |   [12, -70]  |   [-5/4; 5]     |  Up  (Vertex)

QPName = "Far Exterior"

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

alTest = augLagQP_AffineIneq(thisQP, thisConstr, 1, zeros(size(bVec)))
print("Evaluating the Augmented Lagrangian at the starting value of $x0: ")
println(evalAL(alTest, x0))
println("Evaluating the AL gradient: $(evalGradAL(alTest, x0))")
println("Evaluating the AL hessian: $(evalHessAl(alTest, x0))")
