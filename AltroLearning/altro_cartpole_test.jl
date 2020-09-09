# Altro Cartpole Tests

using TrajectoryOptimization
using Altro
import RobotZoo.Cartpole
using StaticArrays, LinearAlgebra

using Plots
pyplot()
using TrajOptPlots
using MeshCat

println("Running the Cartpole example...")

# Make the model
model = Cartpole()
n,m = size(model)
# n is the size of the states ([x, θ, v, ω])
# m is the size of the control (move the cart back and forth along x)

# Trajectory Discretization
N = 151
tf = 5.
dt = tf/(N-1)

# Initial and Final Conditions
x0 = @SVector zeros(n) # Start at rest, pendulum down
xf = @SVector [0, pi, 0, 0]  # Swing pendulum up and end at rest

# LQR Objective Set Up
Q = 1.0e-1*Diagonal(@SVector ones(n))
Qf = 100.0*Diagonal(@SVector ones(n))
R = 1.0*Diagonal(@SVector ones(m))
obj = LQRObjective(Q,R,Qf,xf,N)


# Now we get into the other constraints
# Create Empty ConstraintList
conSet = ConstraintList(n,m,N)

# Control Bounds
u_bnd = 3.0
bnd = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
add_constraint!(conSet, bnd, 1:N-1)

# Goal Constraint
goal = GoalConstraint(xf)
add_constraint!(conSet, goal, N)

# Package the objective and constraints into a "problem" type
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)

# Now, initialize the trajectory
u0 = @SVector fill(0.01,m) # small controls
U0 = [u0 for k = 1:N-1] # vector of the small controls
initial_controls!(prob, U0)
rollout!(prob);

# The last step before solving is the solver options
opts = SolverOptions(
    cost_tolerance_intermediate=1e-2,
    penalty_scaling=10.,
    penalty_initial=1.0
)

altro = ALTROSolver(prob, opts)
set_options!(altro, show_summary=true)
solve!(altro)

X = states(altro)


# Now we want to plot the results
x1 = [r[1] for r in X]
x2 = [r[2] for r in X]

xball = [x1[i] + sin(x2[i]) for i in 1:N]
yball = [-cos(x2[i]) for i in 1:N]

xcart = x1
ycart = zeros(size(x1))

plt = plot(xball, yball, label = "Ball")
plot!(xcart, ycart, label = "Cart")
scatter!([xball[1]], [yball[1]], label = "Initial State")
scatter!([xball[end]], [yball[end]], label = "Final State")

display(plt)

println("\nStarting GIF")

@gif for i in 1:N
    plot(xball[1:i], yball[1:i], label = "Ball")
    plot!(xcart[1:i], ycart[1:i], label = "Cart")
    scatter!([xball[1]], [yball[1]], label = "Initial State")
    scatter!([xball[end]], [yball[end]], label = "Final State")
end

println("Starting pole GIF")

@gif for i in 1:N
    plot([xcart[i], xball[i]], [ycart[i], yball[i]], label = "Pole")
    scatter!([xcart[i]], [ycart[i]], label = "Cart")
    scatter!([xball[1]], [yball[1]], label = "Initial State")
    scatter!([xball[end]], [yball[end]], label = "Final State")
    xlims!(minimum(xball), maximum(xball))
    ylims!(minimum(yball), maximum(yball))
end

println("Starting MeshCat")

vis = Visualizer()
render(vis)

TrajOptPlots.set_mesh!(vis, model)
visualize!(vis, altro);

println("Complete")
