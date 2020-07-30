# Generates a batch of plots

include("plotTrajectory.jl")
include("plotConstraintViolation.jl")
include("plotObjective.jl")
include("trajectoryParsing.jl")


function batchPlot(trajStates, nDim::Int64)
    ptList = [getParseTrajectory(traj, nDim) for traj in trajStates]
    pltTraj = plotTrajPos2D_Multiple(ptList)
    xlabel!("X (km)")
    ylabel!("Y (km)")
    title!("Test Trajectory")
    display(pltTraj)

    pltCV = plotConstraintViolation(cMRocket, trajStates, penaltyStart)
    display(pltCV)
    pltObj = plotObjective(costFun, trajStates)
    display(pltObj)

    # plotSVUTime_Simple(ptList[1])
    plts, pltv, pltu = plotSVUTime_StartEnd(ptList)

    return pltTraj, pltCV, pltObj, plts, pltv, pltu
end


function saveBulk(pltTraj, pltCV, pltObj, plts, pltv, pltu, header::String)
    savefig(pltTraj, header * "Trajectory")
    savefig(pltCV, header * "ConstraintViolation")
    savefig(pltObj, header * "Objective")
    savefig(plts, header * "PositionTime")
    savefig(pltv, header * "VelocityTime")
    savefig(pltu, header * "ControlsTime")
end
