#=

In this file we simulate a simple rocket landing

Units in kg, m, s

=#
include("src\\rocket\\rocket-setup.jl")
include("src\\dynamics\\trajectory-setup.jl")
include("src\\objective\\LQR_objective.jl")
include("src\\constraints\\constraintManager.jl")
include("src\\auglag\\auglag-core.jl")
include("src\\solver\\AL-Primal-Main-Solver.jl")
include("src\\results\\trajectoryParsing.jl")
include("src\\results\\plotTrajectory.jl")

# Based on the Falcon 9
# 549,054 kg (Mass)
# 282 s (Specific Impulse)
# select "y" as the vertical
mass = 549054
isp = 282
grav = [0; -9.81]
deltaTime = 0.1
rocket = rocket_simple(mass, isp, grav, deltaTime)

# in km
# roughly half of the Karman Line (100 km)
rocketStart = [5.0; 50.0; 0.0; -1.0]
rocketEnd = [0.0; 0.0; 0.0; 0.0]#[-5.0; 0.0; 0.0; 0.0]

# Number of time steps to discretize the trajectory
NSteps = 40
# Initialize the trajectory with a line
initTraj = initializeTraj(rocketStart, rocketEnd, NSteps)

# Use a Linear Quadratic Regulator as the cost function
lqrQMat = 10 * Diagonal(I, size(rocketStart, 1))
lqrRMat = 2 * Diagonal(I, Int64(size(rocketStart, 1) / 2))
costFun = makeLQR_TrajSimple(lqrQMat, lqrRMat, NSteps)

ADyn, BDyn = rocketDynamicsFull(rocket, rocketStart, rocketEnd, NSteps)
dynConstraint = AL_AffineEquality(ADyn, BDyn)
lambdaInit = zeros(size(BDyn))

cMRocket = constraintManager([dynConstraint], [lambdaInit])

penaltyStart = 1.0

# Test that the evaluations work
println("\n--------------------------------------------")
println(" Testing constraints are inputted correctly ")
println("--------------------------------------------")
println("Starting constraint violation: ")
println([evalConstraints(cMRocket, initTraj, penaltyStart)])
println("Starting gradient of constraints: ")
println(evalGradConstraints(cMRocket, initTraj, penaltyStart))
println("Starting hessian of constraints: ")
println(evalHessConstraints(cMRocket, initTraj))


# Equiped with the constraint term and the objective term, I now build the
# Augmented Lagrangian
alRocket = augLag(costFun, cMRocket, penaltyStart)

# Test that the evaluations work
println("--------------------------------------------")
println("           Testing AL is Functional         ")
println("--------------------------------------------")
println("Evaluating augmented lagrangian: ")
println([evalAL(alRocket, initTraj)])
println("Evaluating gradient of augmented lagrangian: ")
println(evalGradAL(alRocket, initTraj))
println("Evaluating hessian of augmented lagrangian: ")
println(size(evalHessAl(alRocket, initTraj)))

# Next we select resonable solver parameters
currSolveParams = solverParams(0.1, 0.5,
                                3, 2,
                                10^-4,
                                10, 10^6,
                                2.5, 2, 0.2, 0.2, 0.4)
solParamPrint(currSolveParams)
println()
println()

println("--------------------------------------------")
println("             Beginning Solve                ")
println("--------------------------------------------")

# Solve the Trajectory Optimization problem
trajStates, resArr = ALPrimalNewtonMain(initTraj, alRocket, currSolveParams)

# Get the parsed list of trajectories
ptList = [getParseTrajectory(traj, 2) for traj in trajStates]
plotTrajPos2D_Multiple(ptList)
xlabel!("X")
ylabel!("Y")
title!("Test Trajectory")
