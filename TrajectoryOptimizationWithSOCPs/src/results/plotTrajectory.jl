# Plot Trajectories

include("trajectoryParsing.jl")


using Plots
pyplot()

function plotTrajPos2D_Simple(xList, yList)
    plt = plot(xList, yList, markershape = :circle)
    return plt
end

function plotTrajPos2D_Simple(pt::parseTrajectory)
    xyList = splitDimensions(pt.sList)
    xList = xyList[1]
    yList = xyList[2]
    return plotTrajPos2D_Simple(xList, yList)
end

function plotTrajPos2D_Multiple(ptList)
    plt = plot()

    for pt in ptList
        xyList = splitDimensions(pt.sList)
        xList = xyList[1]
        yList = xyList[2]
        plot!(xList, yList, markershape = :circle)
    end

    return plt
end
