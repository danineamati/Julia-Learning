# using Plots
# pyplot()
# using Pkg
# Pkg.add("Gadfly")
# using Gadfly

hwhj = "Hello World! Hello Julia"

println(hwhj)

function example(x, y)
    return sin.(x) .+ y
end

x0 = range(0, stop=pi, step=0.1)
y0 = example(x0, 2)

# plot(x0, y0)
# gui()

# using Plots
# pyplot()
# plot(rand(5,5), linewidth=2, title="MyPlot2")

using Plots; pyplot()
x=range(-2,stop=2,length=100)
y=range(sqrt(2),stop=2,length=100)
f(x,y) = x*y-x-y+1
plot(x,y,f,st=:surface,camera=(-30,30))

println("Plotting Completed")
