#=
First, we are going to do a normal augmented lagrangian option
"normal" in the sense of Primal and not Primal-Dual

Let's start with the base optimization problem

minimize_x f(x)
subject to ||Ax - b|| ≤ (c'x - d)

Which we write as
||s|| ≤ t
Ax - b = s
c'x - d = t

A is an mxn Matrix
b is an mx1 vector
c is an nx1 vector
d is a real number
x is nx1
s is mx1
t is a real number
Note that a 2-norm is assumed

The base optimization problem is actually

minimize_{x, s, t} f(x)
subject to  ||s|| ≤ t
            Ax - b = s
            c'x - d = t

We group the primals as y = [x; s; t]

The corresponding augmented lagrangian is
minimize_y φ(y) = f(x) + (ρ/2) ||c(y)||_2^2 + λ c(y)

which we can write:
f(x) + (ρ/2) c(y)'c(y) + λ c(y)


------------------------------------
Note that due to the nature of the first inequality, we only want to penalize
when the constraint is violated. (i.e. ||s|| > t)
Thus, we simply use a projection onto the second order cone (s, t) to
determine the violation of the constraint.
----------------------

Note that we solve this problem sequentially. At each iteration, we hold
ρ and λ fixed and solve minimize_x φ(x).
At the end of the iteration,
       λ ← λ + ρ c(x_k*)       - Which is to say update with prior x*
       ρ ← min(ρ * 10, 10^6)   - Which is to say we bound ρ's growth by 10^6

The first order condition for minimize_y φ(y) is
∇_y φ(y) = 0 = ∇f(x) + J(c(y))'(ρ c(y) + λ)
Thus, (ρ c(x) + λ) acts as a "modified lagrange multiplier." Moreover, J(c(y))
is the Jacobian of the constraints.

We now want to use Newton's Method to find x* such that
∇_y φ(y) = 0

Recall that Newton's method for n(x) = 0 is x ← x - n(x)/n'(x)

Thus, we have: y ← y - [H(φ(y))]^-1 ∇φ(y)
Where H is the Hessian Matrix

We now ask: What is H(φ(x))? (The nxn Hessian)
H(φ(x)) = H(f(x)) + ((ρ c(x) + λ) H(c(x)) + ρ ∇c(x) * ∇c(x))
        = H(f(x)) + H((ρ/2) c(y)'c(y) + λ c(y))


=#

using LinearAlgebra

include("QP-Setup-Simple.jl")
include("backtrackLineSearch.jl")
include("constraints.jl")


function newtonStepALP(x0, al::augLagQP_AffineIneq)
    #=
    y ← y - [H(φ(y))]^-1 ∇φ(y)
    returns -[H(φ(y))]^-1 ∇φ(y) and ∇φ(y)
    since -[H(φ(y))]^-1 ∇φ(y) is the step and ∇φ(y) is the residual
    =#
    phiDDinv = inv(evalHessAl(al, x0))
    phiD = evalGradAL(al, x0)
    # Note the negative sign!
    return -phiDDinv * phiD, phiD
end

function newtonMethodLineSearchALP(x0, al::augLagQP_AffineIneq, sp::solverParams,
                                            verbose = false)
    xNewtStates = []
    residNewt = []
    # push!(xNewtStates, x0)
    xCurr = x0

    # For printing
    early = false

    lineSearchObj(x) = evalAL(al, x)
    lineSearchdfdx(x) = evalGradAL(al, x)

    # Now, we run through the iterations
    for i in 1:(sp.maxNewtonSteps)
        # Negative sign addressed above
        (dirNewton, residual) = newtonStepALP(xCurr, al)
        push!(residNewt, residual)

        if verbose
            println("Newton Direction: $dirNewton")
            println("AL: $al")
        end

        # Then get the line search recommendation
        x0LS, stepLS = backtrackLineSearch(xCurr, dirNewton,
                        lineSearchObj, lineSearchdfdx, sp.paramA, sp.paramB)

        if verbose
            println("Recommended Line Search Step: $stepLS")
            println("Expected x = $x0LS ?= $(xCurr + stepLS * dirNewton)")
        end

        push!(xNewtStates, x0LS)

        if norm(xCurr - x0LS, 2) < sp.xTol
            println("Ended from tolerance at $i Newton steps")
            early = true
            break
        else
            xCurr = x0LS
        end

    end

    if !early
        println("Ended from max steps in $(sp.maxNewtonSteps) Newton Steps")
    end

    return xNewtStates, residNewt

end

function ALPrimalNewtonQPmain(x0, al::augLagQP_AffineIneq, sp::solverParams,
                                            verbose = false)

    # (x0, fObj, dfdx, Q, c, A, b, rho, lambda,
    # xtol = 10^-6, maxIters = 5, paramA = 0.1, paramB = 0.5, verbose = false)

    xStates = []
    residuals = []
    push!(xStates, x0)

    for i in 1:(sp.maxOuterIters)

        # Update x at each iteration
        if verbose
            println()
            println("Next Full Update starting at $x0")
        end

        (xNewStates, resAtStates) = newtonMethodLineSearchALP(x0, al, sp, verbose)

        # Take each step in the arrays above and save it to the respective
        # overall arrays
        xStates = [xStates; xNewStates]
        residuals = [residuals; resAtStates]

        # Determine the new lambda and rho
        # λ ← λ + ρ c(x_k*)       - Which is to say update with prior x*
        # ρ ← min(ρ * 10, 10^6)   - Which is to say we bound ρ's growth by 10^6
        xNewest = xNewStates[end]
        APost = getGradC(al.constraints, xNewest)

        if verbose
            println()
            println("APost = $APost")
            println("xStates = $xStates")
            println("xNewest = $xNewest")
            println("All residuals = $residuals")
        end
        lambdaNew = al.lambda + al.rho * (APost * xNewest - al.constraints.b)
        al.lambda = max.(lambdaNew, 0)
        al.rho = clamp(al.rho * sp.penaltyStep, 0, sp.penaltyMax)

        if verbose
            println("New state added")
            println("Lambda Updated: $(al.lambda)")
            println("rho updated: $(al.rho)")
        end

        if norm(xNewest - x0, 2) < sp.xTol
            println("Ended early at $i outer steps")
            break
        else
            x0 = xNewest
        end
    end

    return xStates, residuals
end
