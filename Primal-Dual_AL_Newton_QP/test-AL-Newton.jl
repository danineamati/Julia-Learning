# Test the Augemented Lagrangian Monte Carlo Style

include("augmentedLagranginaMethod.jl")
include("..\\Primal-Dual_IP_Newton_QP\\QP-Setup.jl")

QMat, cVec, AMat, bVec, x0 = QPSetup(true, true)

fObj(x) = (1/2) * x'QMat*x + cVec'x
dfdx(x) = QMat * x + cVec

rhoStart = 1
lambdaStart = zeros(size(AMat, 1))

xStates = ALNewtonQPmain(x0, fObj, dfdx, QMat, cVec, AMat, bVec,
                                rhoStart, lambdaStart)
xVals = [x[1] for x in xStates]
yVals = [x[2] for x in xStates]
plot!(xVals, yVals, markershape = :circle)
