# In this next script we compare the convergence rate of the the following
# solvers:
#   - Primal-Dual Interior Point
#   - Primal-Dual Augmented Lagrangian
#   - Augmented Lagrangian (Primal)

include("..\\Primal-Dual_IP_Newton_QP\\QP-Setup.jl")
include("..\\Primal-Dual_IP_Newton_QP\\primal-dual-IP-Newton-QP.jl")
include("..\\Primal-Dual_AL_Newton_QP\\primal-dual-AL-Newton-QP.jl")
include("..\\Primal-Dual_AL_Newton_QP\\augmentedLagrangianMethod.jl")

println("Complete")
