# Test the Constraint Manager with dynamics

include("..\\src\\rocket\\rocket-setup.jl")
include("..\\src\\dynamics\\trajectory-setup.jl")
include("..\\src\\constraints\\constraintManager.jl")



# Based on the Falcon 9
# 549,054 kg (Mass)
# 282 s (Specific Impulse)
# select "y" as the vertical
mass = 549054
isp = 282
grav = [0; -9.81]
deltaTime = 0.1
rocket = rocket_simple(mass, isp, grav, deltaTime)


# Dynamics code needs initial and final conditions
rocketStart = [5.0; 50.0; 0.0; -1.0]
rocketEnd = [0.0; 0.0; 0.0; 0.0]

# Number of time steps to discretize the trajectory
NSteps = 40

# Initialize the trajectory with a line
initTraj = initializeTraj(rocketStart, rocketEnd, NSteps)

# Build the Dynamics constraints
aMat, gMat = rocketDynamicsStack(rocket, size(grav, 1), NSteps)
dynStack = AL_AffineEquality(aMat, gMat)
lambdaInit = zeros(size(gMat))
cMStack = constraintManager([dynStack], [lambdaInit])

penaltyTest = 10.0

println("Evaluating the Constraints: ")
println([evalConstraints(cMStack, initTraj, penaltyTest)])

println("Evaluate Gradient of the Constraint Terms: ")
println(evalGradConstraints(cMStack, initTraj, penaltyTest))

println("Evaluate Hessian of the Constraint Terms: ")
println(evalHessConstraints(cMStack, initTraj))
