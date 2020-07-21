# We have a similar QP set-up as before, but now we declare instances
# or the key structs.

include("SOCP-Setup-Simple.jl")
include("constraints.jl")
include("montecarlo.jl")


# -------------------------
# Solver Parameters
#=
paramA::Float16         # Used in Line Search, should be [0.01, 0.3]
paramB::Float16         # Used in Line Search, should be [0.1, 0.8]
maxOuterIters::Int32    # Number of Outer Loop iterations
maxNewtonSteps::Int32   # Number of Newton Steps per Outer Loop iterations
xTol::Float64           # When steps are within xTol, loop will stop.
penaltyStep::Float16    # Multiplies the penalty parameter per outer loop
penaltyMax::Float64     # Maximum value of the penalty parameter
=#
# -------------------------
currSolveParams = solverParams(0.1, 0.5, 6, 4, 10^-10, 10, 10^6)
solParamPrint(currSolveParams)

# --------------------------
# Set an example initial starting point
# --------------------------
x0 = [-0.5; 1.25]
# Test 1: [-2; 3]
# Test 2: [-4.5; 3.5]
# Test 3: [-2.5; -0.75]
# Test 4: [-0.5; 1.25]
#[-4.4624; 0.62624] #getPoint([-5,-1],[0,3]) #[-4.6; 2.5] # [-4.75; 2]
#[-0.440932, -0.252654] #[-2; 3]

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
QMat = Symmetric([6 5; 0 8])#Symmetric([0 0; 0 0]) #Symmetric([6 5; 0 8])
println()
print("Q Matrix is Positive Definite: ")
println(isposdef(QMat))

# With Option 1
# For an Interior Point, try [4; -3]
# For an Exterior Point, try [-30; 70]
# With Option 2
# For an Exterior Point, try [12, -70]
pVec = [4; -3]

# Generate the Struct
thisQP = objectiveQP(QMat, pVec)

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

AMat = [4 -5; 3 2] # [4 -5; 3 2] #[4 -5]
bVec = [-10; -9]#[-20; 0] #[-20; 0] #[-20]
cVec = [2; 2]
dVal = -8

# Generate the struct
thisConstr = AL_coneSlack(AMat, bVec, cVec, dVal)

# Now initialize s and t
s0 = [0; 1] #AMat * x0 - bVec #[0; 1]
t0 = 20 #cVec'x0 - dVal #20
println("Other Primals: s = $s0 and t = $t0")

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
print("Evaluating the constraints: ")
println(getNormToProjVals(alcone.constraints, x0, s0, t0))
grad0 = evalGradAL(alcone, y0, true)
println("Evaluating the AL gradient: $(grad0)")
hess0 = evalHessAl(alcone, y0)
println("Evaluating the AL hessian: (Pos. Def.? $(isposdef(hess0)))")
display(hess0)

damp = 10^-2
hessDamp = hess0 + damp * I
print("Evaluating the AL hessian with damping")
println(" (Pos. Def.? $(isposdef(hessDamp)))")
display(hessDamp)
