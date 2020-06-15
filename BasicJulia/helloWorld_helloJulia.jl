using Plots

hwhj = "Hello World! Hello Julia"

println(hwhj)

function example(x, y)
    return sin.(x) .+ y
end

x0 = range(0, stop=5*pi, step=0.1)
y0 = range(0, length=length(x0), step=0.1)

plt = plot3d(1, xlim=(0,5), ylim=(0,5), zlim=(0,5),
                title = "Simple Sine Offset", marker = 2)

@gif for i=1:length(x0)
    push!(plt, x0[i], y0[i], example(x0[i], y0[i]))
end every 10

println("Plotting Completed")
