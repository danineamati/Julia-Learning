@time using Plots
using Colors
using LinearAlgebra

gr()

println("Hello World")

rand(3, 3)

function increment(inputVal)
    return inputVal .+ 1
end

println(increment(5))
println(increment([4, 5, 6]))

someColors = distinguishable_colors(100)

rand(someColors, 3, 3)

numVals = 500

xVals = rand(numVals, 1)
yVals = rand(numVals, 1)

@time s1 = scatter(xVals, yVals)
xVals2 = xVals .^ 10

@time s2 = scatter(xVals2, yVals)
yVals2 = yVals .^ 10

@time s3 = scatter(xVals2, yVals2)

xVals2 = xVals .+ 10
@time s4 = scatter(xVals2, yVals)

plot(s1, s2, s3, s4, layout=(2,2))

numVals = 100

scatList = [s1, s2, s3, s4]

for i in [1:4;]
    xVals = rand(numVals * i, 1)
    yVals = rand(numVals * i, 1)

    scatList[i] = scatter(xVals, yVals)

end

pltShow = plot(scatList[1], scatList[2], scatList[3], scatList[4], layout=(2,2))
gui(pltShow)

methods(heatmap)

v = rand(5, 1)
h1 = heatmap(v * v')

productList = [h1, h1, h1, h1]

for i in [1:4;]
    v = rand(5, 1)
    # Inner Product
    println(v'v)

    # Outer Product
    hNew = heatmap(v * v')
    productList[i] = hNew
end

plt = plot(productList[1], productList[2], productList[3], productList[4], layout = (2, 2))
display(plt)
# Eigen Systems

println(v)
mat = v * v'

eigen(mat)
