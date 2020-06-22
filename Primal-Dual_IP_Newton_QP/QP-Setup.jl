

# QP Set-Up
# By definition, the QP has the following Properties
# minimize_x (1/2) xT Q x + cT x
# subject to Ax ⩽ B
#
# where (all are Real)
# Q is an nxn Symmetric Matrix (that is positive definite for convex)
# x is an nx1 vector
# c is an nx1 vector
# A is an mxn Matrix
# b is an mx1 vector

using LinearAlgebra
using Plots
# gr() # Its a very simple plot
pyplot()

function checkPosDef(Q)
    # To check Positive Definite, we check that the eigenvals are positive
    # and that the determinant is positive.
    eVals = eigvals(Q)
    numRows = size(Q, 1)
    return (eVals > vec(zeros(numRows, 1))) && (det(Q) > 0)
end

function objectiveContours(fObj, xRange = -10:0.01:10, yRange = -2:0.01:10)
    cPlt = contour(xRange, yRange, (x, y) -> fObj([x; y]))
    return cPlt
end

function constraintsPlot(A, bV, x0, xRange = -10:0.01:10, yMin = 2, yMax = 10)
    plt = scatter([x0[1]], [x0[2]])

    for i in 1:size(bV, 1)
        m = -A[i, 1] / A[i, 2]
        b = bV[i] / A[i, 2]
        linFun(x) = m * x + b

        # Determine fill up or down based on intial point
        if x0[2] > linFun(x0[1])
            # The point is above the line, fill up
            yFill = yMax
        else
            yFill = yMin
        end

        plot!(xRange, linFun, fill = (yFill, 0.25, :auto),
                        lw = 2, label = "g$i")
    end

    scatter!([x0[1]], [x0[2]], markersize = 5, label = "Initial pt",
                                markershape = :rect)
    ylims!((yMin, yMax))
    xlabel!("X")
    ylabel!("Y")
    title!("Feasible Region")

    return plt
end

function QPSetup(showContoursObjective = false, showPlotConstraints = false)
    # ---------------------------
    # Objective Function
    # (1/2) xT Q x + cT x
    # Q is an nxn Symmetric Matrix (that is positive definite for convex
    # c is an nx1 vector
    # --------------------------

    QMat = Symmetric([6 5; 0 8])
    print("Q Matrix is Positive Definite: ")
    println(checkPosDef(QMat))

    cVec = [4; -3]

    # Input x as a COLUMN vector (i.e. x = [4; 3])
    fObj(x) = (1/2) * x'QMat*x + cVec'x

    xExample = [5; 3]
    print("Example evaluation of the objective function at $xExample: ")
    println(fObj(xExample))

    if showContoursObjective
        xRange = -10:0.01:10
        yRange = -2:0.01:10
        cPlt = objectiveContours(fObj, xRange, yRange)
        display(cPlt)
    end


    # ---------------------------
    # Constraint Function
    # Ax ≦ b
    # A is an mxn Matrix
    # b is an mx1 vector
    # --------------------------

    # 4x + 5y ≦ 20 and -4x + 5y ≦ 30 and y > -1
    AMat = [4 5; -4 5; 0 -1]
    bVec = [20; 30; 1]

    # --------------------------
    # Set the initial starting point
    # --------------------------

    x0 = [-2.0; 0.5]

    if showPlotConstraints

        println("Beginning Ploting of Constraints")
        xRange = -10:0.1:10

        yMax = 14
        yMin = -3

        plt = constraintsPlot(AMat, bVec, x0, xRange, yMin, yMax)

        display(plt)
        println("Plotting Complete")
    end



    if showContoursObjective && showPlotConstraints
        println("Beginning Ploting of Constraints with Objective")

        xRange = -10:0.01:10
        yMax = 14
        yMin = -3

        yRange = yMin:0.01:yMax
        cPlt = objectiveContours(fObj, xRange, yRange)
        # contour(xRange, yRange, (x, y) -> fObj([x; y]))



        for i in 1:size(bVec, 1)
            m = -AMat[i, 1] / AMat[i, 2]
            b = bVec[i] / AMat[i, 2]
            linFun(x) = m * x + b

            # Determine fill up or down based on intial point
            if x0[2] > linFun(x0[1])
                # The point is above the line, fill up
                yFill = yMax
            else
                yFill = yMin
            end

            plot!(xRange, linFun, fill = (yFill, 0.25, :auto),
                            lw = 2, label = "g$i")
        end

        scatter!([x0[1]], [x0[2]], markersize = 5, label = "Initial pt",
                                    markershape = :rect)
        ylims!((yMin, yMax))
        xlabel!("X")
        ylabel!("Y")
        title!("Feasible Region")

        display(cPlt)
        println("Plotting Complete")

    end


    # Return all key values
    return (QMat, cVec, AMat, bVec, x0)

end
