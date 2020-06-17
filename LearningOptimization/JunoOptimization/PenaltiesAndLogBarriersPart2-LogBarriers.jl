using Plots
using LinearAlgebra
using Optim

include("gradientDescent2D.jl")
include("logBarrier.jl")
include("logBarrierLine.jl")


plotly()
# pyplot()
# gr()

###########################################
# In this Script, I test the log Barrier Method and evaluate it's efficacy
###########################################

getCVal(c, i, n) = c^((i - 1) / (2 * (n - 1)));

function cMat(c, n)
    matA =  zeros(n, n)
    for i in [1:n;]
        matA[i, i] = getCVal(c, i, n)
    end
    return Diagonal(matA)
end

cMatHere = cMat(4, 2)
println(cMatHere)


# First, get the base function f(x)
fsq2C(x) = cMatHere[1, 1] * x[1]^2 + cMatHere[2, 2] * x[2]^2;
fhole2C(x) = 1 - exp(-(fsq2C(x)));

# Second, get the derivative of the base function f(x)
firstDervHole2(x) = [-fhole2C(x)+1 0; 0 -fhole2C(x)+1]
dfhole2(x) = 2 * cMatHere^2 * firstDervHole2(x) * x

# Third, get the sum of the constraint functions
# gCons(x) = [x'x - 1, x[length(x)] + 1 / 4]
# function g1(x)
#     val = x'x -1
#     if val <= 0
#         # Constraint Satisfied
#         return 0
#     else
#         return val
#     end
# end

g1(x) = x'x -1

# function g2(x)
#     val = x[length(x)] + 1/ 4
#     if val <= 0
#         # Constraint Satisfied
#         return 0
#     else
#         return val
#     end
# end

g2(x) = x[length(x)] + 1/4

function sumG(x)
    return g1(x) + g2(x)
end

# Fourth, get the derivative of the constraint functions
dgdx1(x) = 2 * x
dgdx2(x) = [0, 1]
sumdgdx(x) = dgdx1(x) + dgdx2(x)

# sqrPenMethod(xInit, fFun, gFun, dfdx, dgdx, tolx, tolg, maxIter = 10)

xInit1 = [0.5, -0.5]
tolx1 = 0.001
tolg1 = 0.001
maxIt1 = 20



println("Beginning Log Barrier with Line Search")
# logBarrierLine(xInit, fFun, gFunArr, dfdx, dgdxArr, tolx, tolg,
                                    # maxIter = 10, verbose = false)
lbIters = logBarrier(xInit1, fhole2C, [g1, g2], dfhole2, [dgdx1, dgdx2],
                        tolx1, tolg1, maxIt1);

function modLog(x)
    if x < 0
        return -10^6
    else
        return log(x)
    end
end

muTest = 0.1

@time result2 = optimize(x -> fhole2C(x) - muTest * modLog(-g1(x)) -
                        muTest * modLog(-g2(x)), xInit1, BFGS())

function modLog(x)
    if x < 0
        return -10^6
    else
        return log(x)
    end
end

xVals = [r[1] for r in lbIters]
yVals = [r[2] for r in lbIters]

gxrange = -2.4:0.01:2.4
gyrange = -1.5:0.01:1.5

gCons2(x) = [x[1]^2 + x[2]^2 - 1, x[2] + 1 / 4]
fhole2CPlot(x, y) = fhole2C([x, y])

# gCons2Plot1Satisfied(x, y) = gCons2([x, y])[1] < 0
# gCons2Plot2Satisfied(x, y) = gCons2([x, y])[2] < 0

# plt2 = contour(gxrange, gyrange, gCons2Plot2Satisfied, nlevels = 25, fill = true, aspectratio = :equal, seriesalpha = 0.1)
# plt1 = contour!(gxrange, gyrange, gCons2Plot1Satisfied, nlevels = 25, fill = true, aspectratio = :equal, seriesalpha = 0.1)

c1Trace = contour(gxrange, gyrange, (x,y) -> -modLog(-g1([x, y])) - modLog(-g2([x, y])), nlevels = 25, fill = true,
                aspectratio = :equal, seriesalpha = 0.2, clims=(-1.0,10.0))

baseFunc = contour!(gxrange, gyrange, fhole2CPlot, nlevels = 25, fill = true, aspectratio = :equal, seriesalpha = 0.2, clims=(-1.0,10.0))

plot!(xVals, yVals)
scatter!(xVals, yVals)
scatter!([xVals[1], xVals[length(xVals)]], [yVals[1], yVals[length(yVals)]])
scatter!([result2.minimizer[1]],[result2.minimizer[2]])

xlabel!("x")
ylabel!("y")
title!("Constraint, Base Function, Initial Guess")

display(c1Trace)

showRegionsSeparate = false
showOverlayed = false
showSummed = false

if showRegionsSeparate
    plt1 = contour(gxrange, gyrange, (x,y) -> -modLog(-g1([x, y])), nlevels = 25, fill = true,
                aspectratio = :equal, seriesalpha = 0.1, clims=(-1.0,10.0))
    display(plt1)

    plt2 = contour(gxrange, gyrange, (x,y) -> -modLog(-g2([x, y])), nlevels = 25, fill = true,
                    aspectratio = :equal, seriesalpha = 0.1, clims=(-1.0,10.0))
    display(plt2)
end

if showOverlayed
    plt1 = contour(gxrange, gyrange, (x,y) -> -modLog(-g1([x, y])), nlevels = 25, fill = true,
                aspectratio = :equal, seriesalpha = 0.2, clims=(-1.0,10.0))
    contour!(gxrange, gyrange, (x,y) -> -modLog(-g2([x, y])), nlevels = 25, fill = true,
                    aspectratio = :equal, seriesalpha = 0.1, clims=(-1.0,10.0))
    display(plt1)
end

if showSummed
    plt1 = contour(gxrange, gyrange, (x,y) -> -modLog(-g1([x, y])) - modLog(-g2([x, y])), nlevels = 25, fill = true,
                aspectratio = :equal, seriesalpha = 0.2, clims=(-1.0,10.0))
    display(plt1)
end

# Format the constraints with base fun
function sumfglogTest(x, mu)
    val = fhole2C(x)

    gFunArrTest = [g1, g2]

    for i in 1:length(x)
        valg = mu * modLog(-gFunArrTest[i](x))
        val -= valg
    end
    return val
end

muTest = [1, 0.1, 0.01]
climMax = 5.0
numlvls = 25

gyrange2 = -1.5:0.01:0.0

c1 = contour(gxrange, gyrange2, (x,y) -> sumfglogTest([x,y], muTest[1]), nlevels = numlvls, fill = true,
                seriesalpha = 0.3, clims=(-1.0,climMax), title = "mu = " * string(muTest[1]))
c2 = contour(gxrange, gyrange2, (x,y) -> sumfglogTest([x,y], muTest[2]), nlevels = numlvls, fill = true,
                seriesalpha = 0.3, clims=(-1.0,climMax), title = "mu = " * string(muTest[2]))
c3 = contour(gxrange, gyrange2, (x,y) -> sumfglogTest([x,y], muTest[3]), nlevels = numlvls, fill = true,
                seriesalpha = 0.3, clims=(-1.0,climMax), title = "mu = " * string(muTest[3]))

pltcAll = plot(c1, c2, c3, layout = (1, 3))
display(pltcAll)
