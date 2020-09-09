# Altro Quadrotor Tests

using RobotDynamics, Rotations
using TrajectoryOptimization, Altro
using StaticArrays, LinearAlgebra
import RobotZoo.Quadrotor

# Make the model
model = Quadrotor()
n,m = size(model)
#=
n is the size of the states
(13:
    x = 3D position -> 3 values,
    q = 3D orientation as quaternion -> 4 values,
    v = 3D velocity -> 3 values,
    ω = 3D angular velocity -> 3 values
)

m is the size of the control
(4:
    4 rotors
)
=#

# Set up discretization
N = 101                # number of knot points
tf = 5.0               # total time (sec)
dt = tf/(N-1)          # time step (sec)

# Initial and Final Conditions
x0_pos = SA[0, -10, 1.]
xf_pos = SA[0, +10, 1.]
x0 = RobotDynamics.build_state(
                        model, x0_pos, UnitQuaternion(I), zeros(3), zeros(3))
xf = RobotDynamics.build_state(
                        model, xf_pos, UnitQuaternion(I), zeros(3), zeros(3));

# Initialize with a hover trajectory at the initial position.
u0 = @SVector fill(0.5*model.mass/m, m)
U_hover = [copy(u0) for k = 1:N-1]; # initial hovering control trajectory

# Set up the cost function
# Waypoints make a "zig-zag" trajectory
wpts = [SA[+10, 0, 1.],
        SA[-10, 0, 1.],
        xf_pos]
wtimes = [33, 66, 101]   # in knot points

# Set up nominal costs
Q = Diagonal(RobotDynamics.fill_state(model, 1e-5, 1e-5, 1e-3, 1e-3))
R = Diagonal(@SVector fill(1e-4, 4))
q_nom = UnitQuaternion(I)
v_nom = zeros(3)
ω_nom = zeros(3)
x_nom = RobotDynamics.build_state(model, zeros(3), q_nom, v_nom, ω_nom)
cost_nom = LQRCost(Q, R, x_nom)

# Set up waypoint costs
Qw_diag = RobotDynamics.fill_state(model, 1e3,1,1,1)
Qf_diag = RobotDynamics.fill_state(model, 10., 100, 10, 10)
costs = map(1:length(wpts)) do i
    r = wpts[i] # Current Waypoint
    xg = RobotDynamics.build_state(model, r, q_nom, v_nom, ω_nom)
    if wtimes[i] == N
        # Final State
        Q = Diagonal(Qf_diag)
    else
        # Waypoint State
        Q = Diagonal(1e-3*Qw_diag)
    end

    LQRCost(Q, R, xg)
end

# Build Objective
costs_all = map(1:N) do k
    i = findfirst(x->(x ≥ k), wtimes)
    if k ∈ wtimes
        costs[i]
    else
        cost_nom
    end
end
obj = Objective(costs_all);


# Set up the constraints
conSet = ConstraintList(n,m,N)
bnd = BoundConstraint(n,m, u_min=0.0, u_max=12.0) # Max and Min Propeller
add_constraint!(conSet, bnd, 1:N-1)

# Put it all together as a problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
initial_controls!(prob, U_hover)
rollout!(prob);

opts = SolverOptions(
    penalty_scaling=100.,
    penalty_initial=0.1,
)

println("Preparing Solve")

solver = ALTROSolver(prob, opts)
set_options!(solver, show_summary=true)
solve!(solver)
println("Cost: ", cost(solver))
println("Constraint violation: ", max_violation(solver))
println("Iterations: ", iterations(solver))

# Lastly we visualize the result
using TrajOptPlots
using MeshCat
using Plots

vis = Visualizer()
render(vis)

TrajOptPlots.set_mesh!(vis, model)
visualize!(vis, solver);

println("Complete")
