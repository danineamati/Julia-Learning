# In this next script we compare the convergence rate of the the following
# solvers:
#   - Primal-Dual Interior Point
#   - Primal-Dual Augmented Lagrangian
#   - Augmented Lagrangian (Primal)

include("..\\Primal-Dual_IP_Newton_QP\\QP-Setup.jl")
include("..\\Primal-Dual_IP_Newton_QP\\primal-dual-IP-Newton-QP.jl")
include("..\\Primal-Dual_AL_Newton_QP\\primal-dual-AL-Newton-QP.jl")
include("..\\Primal-Dual_AL_Newton_QP\\augmentedLagrangianMethod.jl")

using LinearAlgebra, CSV, DataFrames

# ----------------------------
# First is setup
# ----------------------------

println()
println("Setting Up the QP")
QMat, cVec, AMat, bVec, x0 = QPSetup(true, true)

fObj(x) = (1/2) * x'QMat*x + cVec'x
dfdx(x) = QMat * x + cVec

# For the Interior Point Method
mu = 1

# For the Augmented Lagrangian (including Primal-Dual)
rho = 1
nu = zeros(size(AMat, 1))
lambda = ones(size(AMat, 1))

# φ(x) = f(x) + (ρ/2) c(x)'c(x) + λ c(x)
phiFun(x) = fObj(x) + (rho / 2) * cPlus(AMat, x, bVec)'cPlus(AMat, x, bVec) +
                lambda' * cPlus(AMat, x, bVec)
dPhidx(x) = getQPgradPhiAL(x, QMat, cVec, AMat, bVec, rho, lambda)

println("Objective: f(x) = (1/2) x'Qx + c'x")
println("Q = $QMat")
println("c = $cVec")
println("Constraints: Ax ≤ b")
println("A = $AMat")
println("b = $bVec")
println("Initial Starting Point: $x0")

xStates = pdALNewtonQPmain(QMat, cVec, AMat, bVec, x0, lambda, rho, nu,
                fObj, dfdx, 0.1, 0.5, false)

CSV.write("testSave.csv", DataFrame(xStates), writeheader = false)

println("Complete")
