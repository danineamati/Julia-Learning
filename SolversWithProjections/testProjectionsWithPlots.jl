# Test the projections with plots

using Plots
gr()

include("projections.jl")

# First, we plot a line and the distance from the line
xRange = -10:0.1:10
yRange = -8:0.1:8
vecTest = [5; 4]
lineTest(x) = (vecTest[2]/vecTest[1]) * x

distanceSqrLine(x, y) = norm(orthoProjLine([x; y], vecTest))^2

plt = plot(xRange, lineTest, label = "Line Constraint")
contour!(xRange, yRange, distanceSqrLine, label = "Sqr Distance to Line")
title!("Squared of Distance to Line")
xlabel!("X")
ylabel!("Y")
display(plt)
