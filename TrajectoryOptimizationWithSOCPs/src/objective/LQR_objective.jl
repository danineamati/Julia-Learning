# Linear Quadratic Regulator Objective Functions

using SparseArrays

include("QP_Linear_objectives.jl")

abstract type objectiveLQR_abstract <: objectiveFunc end

"""
    LQR_simple

A simple LQR struct where `Q` and `R` are constant for all `x` and `u`.
"""
struct LQR_simple <: objectiveLQR_abstract
    Q
    R
end


#####################################
#  LQR -> ObjectiveQP Constructors  #
#####################################
@doc raw"""
    makeLQR_TrajSimple(Q, R, NSteps::Int64)

Pseudo-constructor for an LQR objective function (which is really an
[`objectiveQ`](@ref) objective function).

An LQR struct for a trajectory is one that holds the sparse matrix matching:

```math
B_{QR} = [Q 0 0 0; 0 R 0 0; 0 0 Q 0; 0 0 0 R]
```

Where size corresponds to the size of the trajectory (or trajectory horizon).

```math
f(y) = \frac{1}{2} \ y^{\top} * B_{QR} * y
```

See also: [`LQR_simple`](@ref)
"""
function makeLQR_TrajSimple(Q, R, NSteps::Int64)
    QSize = size(Q, 1)
    RSize = size(R, 1)
    totSize = (QSize + RSize) * NSteps + QSize

    # Initialize and empty sparse array
    QRFull = spzeros(totSize, totSize)

    rStart = 1

    for k in 1:NSteps
        rEnd = rStart + QSize - 1
        QRFull[rStart:rEnd, rStart:rEnd] = Q

        rStart += QSize
        rEnd = rStart + RSize - 1

        QRFull[rStart:rEnd, rStart:rEnd] = R

        rStart += RSize
    end
    rEnd = rStart + QSize - 1
    QRFull[rStart:rEnd, rStart:rEnd] = Q

    return objectiveQ(QRFull)
end

function makeLQR_TrajSimple(lqr::LQR_simple, NSteps::Int64)
    return makeLQR_TrajSimple(lqr.Q, lqr.R, NSteps)
end
