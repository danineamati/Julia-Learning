# Test the projections with plots

using Plots
gr()

include("projections.jl")


# First, we plot a line and the squared distance from the line
xRange = -10:0.1:10
yRange = -8:0.1:8
vecTest = [5; 4]
lineTest(x) = (vecTest[2]/vecTest[1]) * x

# distanceSqrLine(x, y) = norm(orthoProjLine([x; y], vecTest))^2
distanceSqrLine(x, y) = sqrDistEuclid([x; y], pt -> projLine(pt, vecTest))

plt = plot(xRange, lineTest, label = "Line Constraint")
contour!(xRange, yRange, distanceSqrLine, label = "Sqr Distance to Line")
title!("Square of Distance to Line")
xlabel!("X")
ylabel!("Y")
display(plt)
savefig(plt, "projLineFromVec")

# Second, we plot the distance from the positive orthant
distSqrPosOrth(x, y) = sqrDistEuclid([x; y], pt -> projPosOrth(pt))

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

plt2 = plot(rectangle(10,8,0,0), opacity = 0.25, label = "Positive Orthant")
contour!(xRange, yRange, distSqrPosOrth, label = "Sqr Distance to Line")
title!("Square of Distance to Positive Orthant")
xlabel!("X")
ylabel!("Y")
display(plt2)
savefig(plt2, "projPosOrth")

# Third, we plot the squared distance from an affine equality constraint
# a'x = b -> a1 x + a2 y = b
# So y = (- a1 / a2) * x + (b / a2)
# Note that in the form a'x = b,
# a is the "normal vector to the hyperplane"
# b is the "offset"
aTest = [5; 4]
bTest = 4 * aTest[2]

xRange = -10:0.1:10
yRange = -9:0.1:17

affineTest(x) = (-aTest[1]/aTest[2]) * x + bTest / aTest[2]
distSqrAffine(x, y) = sqrDistEuclid([x; y],
                            pt -> projAffineIneq(aTest, bTest, [x; y]))

plt3 = plot(xRange, affineTest, label = "Affine Constraint",
                        fill = (minimum(yRange), 0.25, :auto))
contour!(xRange, yRange, distSqrAffine, label = "Sqr Distance to Line")
title!("Square of Distance to Affine Inequality Constraint")
xlabel!("X")
ylabel!("Y")
display(plt3)
savefig(plt3, "projAffineIneq")

sTest = 6
distSqr2Cone(x, y) = sqrDistEuclid([x; y],
                            pt -> projSecondOrderCone([x; y], sTest)[1:2])

xRange = -15:0.1:15
yRange = -15:0.1:15
# plt4 = contour(xRange, yRange, (x, y) -> norm([x; y]))
plt4 = contour(xRange, yRange, distSqr2Cone)
title!("Square of Distance to Cone at s = $sTest slice")
xlabel!("X")
ylabel!("Z")
display(plt4)
savefig(plt4, "proj2ConeSimple")

plt5 = surface(xRange, yRange, (x, y) -> norm([x; y]),
                    xlabel = "X", ylabel = "Y", zlabel = "t")
title!("Second Order Cone")
display(plt5)
savefig(plt5, "proj2ConeContours")

xRange = -15:0.1:15
yRange = -5:0.1:15
# Trying the plot in x ∈ R, t ∈ R
plt6 = contour(xRange, yRange, (x, y) -> projSecondOrderCone(x, y)[1])
plot!(xRange, x -> abs(x), linecolor = :black, label = "Cone Surface")
xlabel!("X")
ylabel!("t")
title!("1D 2nd Order Cone Projections")
display(plt6)
savefig(plt6, "proj2Cone1D")

# Repeat, but now with an affine input
xRange = -50:0.1:10
yRange = -5:0.1:15
aTest = 1/5
bTest = -6
plt7 = contour(xRange, yRange,
            (x, y) -> projSecondOrderCone(aTest * x - bTest, y)[1])
plot!(xRange, x -> abs(aTest * x - bTest), linecolor = :black, label = "Cone Surface")
xlabel!("X")
ylabel!("t")
title!("1D 2nd Order Cone Projections from Affine (Ax - b)")
display(plt7)
savefig(plt7, "proj2Cone1DAffine.png")
