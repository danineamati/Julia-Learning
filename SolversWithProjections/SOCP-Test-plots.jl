# Cone Plots

include("constraints.jl")
include("projections.jl")

using Plots
pyplot()

# This is our constraint
dconeT2DSimp = AL_pCone([2 1/3; 1/3 3/4], [0; 0], [0; 0], -8, 2)

# This is our grid for the initial mapping of x → Ax-b
xyGrid = -14:2:14

# This stores the results from the initial mapping
vxList = []
vyList = []
sList = []

# This stores teh reuslts from the projection
projvxList = []
projvyList = []
projsList = []

# We loop through every point in the grid and calculate the results of the
# initial mapping and the projection.
for x in xyGrid
    for y in xyGrid
        v = dconeT2DSimp.A * [x; y] - dconeT2DSimp.b
        push!(vxList, v[1])
        push!(vyList, v[2])
        s = dconeT2DSimp.c' * [x; y] - dconeT2DSimp.d
        push!(sList, s)

        xproj, proj = getProjVecs(dconeT2DSimp, [x; y])
        push!(projvxList, proj[1])
        push!(projvyList, proj[2])
        push!(projsList, proj[3])
    end
end

# First, make a 3D Plot of the results
# pltVS = scatter3d(vxList, vyList, sList, xlabel = "v1", ylabel = "v2",
#                     zlabel = "s", label = "Base Plot")
# scatter3d!(projvxList, projvyList, projsList, label = "Projection")
# title!("Second Order Cone Visualization")
# display(pltVS)
# savefig(pltVS, "vs3DVisual")
#
# # Top View
# pltV = scatter(vxList, vyList, xlabel = "v1", ylabel = "v2",
#                         label = "Base Plot")
# scatter!(projvxList, projvyList, label = "Projection")
# xlabel!("v1")
# ylabel!("v2")
# title!("Second Order Cone Visualization of the v-Plane")
# display(pltV)
# savefig(pltV, "vsVVplaneVisual")
#
# # Side View 1
# pltS1 = scatter(vxList, sList, label = "Base Plot")
# scatter!(projvxList, projsList, label = "Projection")
# xlabel!("v1")
# ylabel!("s")
# title!("Second Order Cone Visualization of the v1s-Plane")
# display(pltS1)
# savefig(pltS1, "vsV1SplaneVisual")
#
# # Side View 2
# pltS2 = scatter(vyList, sList, label = "Base Plot")
# scatter!(projvyList, projsList, label = "Projection")
# xlabel!("v2")
# ylabel!("s")
# title!("Second Order Cone Visualization of the v2s-Plane")
# display(pltS2)
# savefig(pltS2, "vsV2SplaneVisual")


# Now, we address a different question which is the value of the constraint
# and the violation results

xExtended = []
yExtended = []
violList = []
valueList = []
for x in xyGrid
    for y in xyGrid
        push!(xExtended, x)
        push!(yExtended, y)
        push!(violList, getNormToProjVals(dconeT2DSimp, [x; y]))
        push!(valueList, getRaw(dconeT2DSimp, [x; y]))
    end
end

pltVio = scatter3d(xExtended, yExtended, violList, label = "Violation")
xlabel!("x")
ylabel!("y")
title!("Calculated Constraint Violation")
display(pltVio)
savefig(pltVio, "vioPlot3D")

pltVio2D = scatter(xExtended, violList, label = "Violation")
xlabel!("x")
ylabel!("Violation")
title!("Calculated Constraint Violation")
display(pltVio2D)
savefig(pltVio2D, "vioPlot2D")

pltVioVal = scatter3d(xExtended, yExtended, violList, label = "Violation")
scatter3d!(xExtended, yExtended, valueList, label = "Constraint Value")
xlabel!("x")
ylabel!("y")
title!("Comparison of Calculated Constraint Violation and Value")
display(pltVioVal)
savefig(pltVioVal, "viovalPlot3D")

pltVioVal2D = scatter(xExtended, violList, label = "Violation")
scatter!(xExtended, valueList, label = "Constraint Value")
xlabel!("x")
ylabel!("Violation or Value")
title!("Comparison of Calculated Constraint Violation and Value")
display(pltVioVal2D)
savefig(pltVioVal2D, "viovalPlot2D")
