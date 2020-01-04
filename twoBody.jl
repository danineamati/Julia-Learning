# Two-Body Problem Test
#=
We have the general ODE r'' = - (G m / |r|^3) * r
where r is the radius vector.

We can convert this to a system of two first-order ODEs:
r' = v
v' = - (Gm / |r|^3) * r
where v represents the velocity.

Here, we will solve a unitless two-body problem and hence
keep "Gm = 1"

I will also use https://janus.astro.umd.edu/orbits/elements/convertframe.html
to find our initial conditions (for the sake of ease).

I will make a function that does the above later.

=#

using LinearAlgebra
using DifferentialEquations
using Plots
pyplot()

gm = 1.0
period(a) = 2 * π * sqrt((a^3) / gm)

function twobody!(du, u, p, t)
    #= u represents the state vector
    u[1:3] = radius vector
    u[4:6] = velocity vector

    du[1:3] = r'
    du[4:6] = v'
    =#
    rMag = sqrt.(u[1].^2 + u[2].^2 + u[3].^2)
    coeff = gm ./ (rMag.^3)
    # du[1:3] = u[4:6]
    # du[4:6] = - coeff .* u[1:3]
    du[1] = u[4]
    du[2] = u[5]
    du[3] = u[6]
    du[4] = - coeff .* u[1]
    du[5] = - coeff .* u[2]
    du[6] = - coeff .* u[3]
end

function lorenz!(du,u,p,t)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end

function example!(du, u, p, t)
    radMag = sqrt.(u[1].^2 + u[2].^2)
    du[1] = - (gm / (abs(u[1]).^3)) * u[1]
    du[2] = u[2]
end

# Now we set up the ODE
# We the initial conditions as [x, y, z, dx, dy, dz]
uInit = [0.99; 0.0; 0.0; 0.0; 0.87472940; 0.50502525]
# uInit2 = [0.99; 0.5];

# u0 = [1.0;0.0;0.0]
# tspan = (0.0,100.0)
# prob = ODEProblem(lorenz!,u0,tspan)
# sol = solve(prob)

# We now want the interval of interest (one full period)
aSemimajor = 1.0
tspan = (0.0, period(aSemimajor))
# tspan = (0.0, 0.4)

# Lastly we make the ODE itself
prob = ODEProblem(twobody!, uInit, tspan)
println("ODE Created")

# prob = ODEProblem(example!, uInit2, tspan)
sol = solve(prob, reltol=1e-10, abstol=1e-10)

# plot(sol,linewidth=2,xaxis="t",label=["x" "y" "z" "vx" "vy" "vz"],layout=(2,1))

# USE THIS:
plot(sol, vars=(1,2,3))

println("Solved the ODE, beginning plotting")

plt = plot3d(1, xlim=(-2,2), ylim=(-2,2), zlim=(-2,2),
                title = "Two Body Problem", marker = 2)

@gif for i=1:length(sol)
    println("Step $i")
    push!(plt, sol.u[i][1], sol.u[i][2], sol.u[i][3])
end every 10

println("Plotting Complete")
