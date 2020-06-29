# In this next script we compare the convergence rate of the the following
# solvers:
#   - Primal-Dual Interior Point
#   - Primal-Dual Augmented Lagrangian
#   - Augmented Lagrangian (Primal)

include("..\\Primal-Dual_IP_Newton_QP\\QP-Setup.jl")
include("..\\Primal-Dual_IP_Newton_QP\\primal-dual-IP-Newton-QP-2.jl")
# include("..\\Primal-Dual_IP_Newton_QP\\primal-dual-IP-Newton-QP.jl")
include("..\\Primal-Dual_AL_Newton_QP\\primal-dual-AL-Newton-QP.jl")
include("..\\Primal-Dual_AL_Newton_QP\\augmentedLagrangianMethod.jl")

using LinearAlgebra, CSV, DataFrames, Plots
pyplot()

# ---------------------------
# Functions for later calculations
# ---------------------------

function calcPerpNormResiduals(xArr)
    xs = [x[1] for x in xArr]
    ys = [x[2] for x in xArr]

    # Notice "Magic Numbers" of the correct value to prevent storing it as a
    # variable (for now. i.e. will be changed)
    # [-5/4; 5] - UP configuration
    # [-47/23; 38/23] - Interior configuration
    # [35/6; -1] - Far Exterior configuration
    res = [norm([xs[ind]; ys[ind]] - [-5/4; 5])^2 for ind in 1:size(xs,1)]
    return res
end

# ----------------------------
# Before we start, we run the is setup
# ----------------------------

println()
println("Setting Up the QP")
QMat, cVec, AMat, bVec, x0 = QPSetup(false, false)

fObj(x) = (1/2) * x'QMat*x + cVec'x
dfdx(x) = QMat * x + cVec

# ------------------------
# Change the initial Point here!
# -----------------------

x0 = [1.0; 2.0]

# For the Interior Point Method
mu = 1

# For the Augmented Lagrangian (Primal-Dual)
rho = 1
nu = zeros(size(AMat, 1))
lambda = zeros(size(AMat, 1))

# For general Line Search
paramA = 0.1
paramB = 0.5

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

# -----------------
# First we run the Primal-Dual Interior Point
# -----------------

println("Starting Primal-Dual with Interior Point")
hStatesPDIP = pdIPNewtonQPmain(QMat, cVec, AMat, bVec, x0, lambda, mu,
                                fObj, dfdx)
resPDIP = calcPerpNormResiduals(hStatesPDIP)
println("Completed Primal-Dual with Interior Point")

# -----------------
# Second we run the Primal-Dual Augmented Lagrangian
# -----------------

println("Starting Primal-Dual with Augmented Lagrangian")
hStatesPDAL = pdALNewtonQPmain(QMat, cVec, AMat, bVec, x0, lambda, rho, nu,
                fObj, dfdx, paramA, paramB, false)
resPDAL = calcPerpNormResiduals(hStatesPDAL)
println("Completed Primal-Dual with Augmented Lagrangian")

# -----------------
# Third we run the Augmented Lagrangian (Primal)
# -----------------
println("Starting Augmented Lagrangian (Primal)")
rhoStart = 1
lambdaStart = ones(size(AMat, 1))

xStatesALP = ALNewtonQPmain(x0, fObj, dfdx, QMat, cVec, AMat, bVec,
                                rhoStart, lambdaStart)
resALP = calcPerpNormResiduals(xStatesALP)
println("Completed Augemented Lagrangian (Primal)")

# ------------------
# Now we want to plot the result
# ------------------

plt = plot(resPDIP, yaxis = :log, label = "PD Interior Point")
plot!(resPDAL, yaxis = :log, label = "PD Augmented Lagrangian")
plot!(resALP, yaxis = :log, label = "Augemented Lagrangian Primal")

xlabel!("Recorded Point in Solver")
ylabel!("Square of Norm Residuals")
title!("Comparison of three Solvers")

savefig(plt, "solverComparison14_Fix-Up")

display(plt)

# CSV.write("testSave.csv", DataFrame(xStates), writeheader = false)

println("\n * Complete *")
