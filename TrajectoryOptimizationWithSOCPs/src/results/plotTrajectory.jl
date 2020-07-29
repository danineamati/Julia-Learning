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

    for (ind, pt) in enumerate(ptList)
        xyList = splitDimensions(pt.sList)
        xList = xyList[1]
        yList = xyList[2]
        plot!(xList, yList, markershape = :circle, label = "Traj $ind")
    end

    return plt
end

function plotSVUTime_Simple(sList, vList, uList)

    # Variables needed for later
    nDim = size(sList[1], 1)
    sxyzLabels = ["x", "y", "z"]
    vxyzLabels = ["vx", "vy", "vz"]
    uxyzLabels = ["ux", "uy", "uz"]

    # Plot the Positions
    plt_s = plot()
    xyzList = splitDimensions(sList)

    for d in 1:nDim
        plot!(xyzList[d], markershape = :circle, label = sxyzLabels[d])
    end
    title!("Change in position over time")
    xlabel!("Time")
    ylabel!("Position")
    display(plt_s)

    # Plot the Velocities
    plt_v = plot()
    vxyzList = splitDimensions(vList)

    for d in 1:nDim
        plot!(vxyzList[d], markershape = :circle, label = vxyzLabels[d])
    end
    title!("Change in velocity over time")
    xlabel!("Time")
    ylabel!("Velocity")
    display(plt_v)

    # Plot the Controls
    plt_u = plot()
    uxyzList = splitDimensions(uList)

    for d in 1:nDim
        plot!(uxyzList[d], markershape = :circle, label = uxyzLabels[d])
    end
    title!("Change in thrust over time")
    xlabel!("Time")
    ylabel!("Contol (Thrust)")
    display(plt_u)

    return  plt_s, plt_v, plt_u
end

function plotSVUTime_Simple(pt::parseTrajectory)
    return plotSVUTime_Simple(pt.sList, pt.vList, pt.uList)
end
