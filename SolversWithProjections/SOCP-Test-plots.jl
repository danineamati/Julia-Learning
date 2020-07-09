# Cone Plots

include("constraints.jl")
include("projections.jl")

using Plots
pyplot()


dconeT2DSimp = AL_pCone([2 1/3; 1/3 3/4], [0; 0], [0; 0], -8, 2)

xyGrid = -14:2:14

vxList = []
vyList = []
sList = []

projvxList = []
projvyList = []
projsList = []

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

pltVS = scatter3d(vxList, vyList, sList, xlabel = "v1", ylabel = "v2",
                    zlabel = "s", label = "Base Plot")
scatter3d!(projvxList, projvyList, projsList, label = "Projection")
title!("Second Order Cone Visualization")
display(pltVS)
savefig(pltVS, "vs3DVisual")

pltV = scatter(vxList, vyList, xlabel = "v1", ylabel = "v2",
                        label = "Base Plot")
scatter!(projvxList, projvyList, label = "Projection")
xlabel!("v1")
ylabel!("v2")
title!("Second Order Cone Visualization of the v-Plane")
display(pltV)
savefig(pltV, "vsVVplaneVisual")

pltS1 = scatter(vxList, sList, label = "Base Plot")
scatter!(projvxList, projsList, label = "Projection")
xlabel!("v1")
ylabel!("s")
title!("Second Order Cone Visualization of the v1s-Plane")
display(pltS1)
savefig(pltS1, "vsV1SplaneVisual")

pltS2 = scatter(vyList, sList, label = "Base Plot")
scatter!(projvyList, projsList, label = "Projection")
xlabel!("v2")
ylabel!("s")
title!("Second Order Cone Visualization of the v2s-Plane")
display(pltS2)
savefig(pltS2, "vsV2SplaneVisual")
