#=

In this file we simulate a simple rocket landing

Units in kg, m, s

=#
include("src\\rocket\\rocket-setup.jl")
include("src\\dynamics\\trajectory-setup.jl")
include("src\\objective\\LQR_objective.jl")
include("src\\constraints\\constraintManager.jl")


# Based on the Falcon 9
# 549,054 kg (Mass)
# 282 s (Specific Impulse)
# select "y" as the vertical
grav = [0; -9.81]
rocket = rocket_simple(549054, 282, grav)

# in km
# roughly half of the Karman Line (100 km)
rocketStart = [5.0; 50.0; 0.0; -1.0]
rocketEnd = [-5.0; 0.0; 0.0; 0.0]

# Number of time steps to discretize the trajectory
NSteps = 10
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

println("Starting constraint violation: ")
println(evalConstraints(cMRocket, initTraj, penaltyStart))
