# First, we are going to do a normal augmented lagrangian option
# "normal" in the sense of not Primal-Dual
#
# Let's start with the base optimization problem
#
# minimize_x f(x)
# subject to c(x) ≤ 0
#
# The corresponding augmented lagrangian is
# minimize_x φ(x) = f(x) + (ρ/2) ||c(x)||_2^2 + λ c(x)
#
# which we can write:
# f(x) + (ρ/2) c(x)'c(x) + λ c(x)
#
#
# ------------------------------------
# Note that due to the nature of the inequality, we only want to penalize
# when the constraint is violated. (i.e. c(x) > 0)
# Thus, we define a c_+(x) = max(c(x), 0). In this formulation, we need a
# piecewise definition of ∇c_+(x):
# If c(x) > 0, then ∇c_+(x) = ∇c(x)
# Else, ∇c_+(x) = 0
#
# Now, we mix this with the quadratic part, i.e. ||c(x)||_2^2 = c(x)'c(x)
# We have c_+(x)'c_+(x), so
# If c(x) > 0, then ∇[c_+(x)'c_+(x)] = ∇[c(x)'c(x)] = 2 c(x) ∇c(x)
# Else, ∇[c_+(x)'c_+(x)] = 0
#
# Notice that the gradient is continuous since at the transition point, since
# c(x) = 0 at the transition (as specified by the piecewise)
#
# Lastly, we take one more gradient, (ρ c(x) + λ) ∇^2c(x) + ρ ∇c(x) * ∇c(x)
# If c(x) > 0, then we have (ρ c(x) + λ) ∇^2c(x) + ρ ∇c(x) * ∇c(x)
# Else, (ρ c_+(x) + λ) ∇^2c_+(x) + ρ ∇c_+(x) * ∇c_+(x) = 0
#
# Notice that the hessian is not continuous. At the transition point, c(x) = 0
# which yields λ ∇^2c(x) + ρ ∇c(x) * ∇c(x) ≠ 0
# ----------------------
#
# Note that we solve this problem sequentially. At each iteration, we hold
# ρ and λ fixed and solve minimize_x φ(x).
# At the end of the iteration,
#        λ ← λ + ρ c(x_k*)       - Which is to say update with prior x*
#        ρ ← min(ρ * 10, 10^6)   - Which is to say we bound ρ's growth by 10^6
#
# The first order condition for minimize_x φ(x) is
# ∇_x φ(x) = 0 = ∇f(x) + (ρ c(x) + λ) ∇c(x)
# Thus, (ρ c(x) + λ) acts as a "modified lagrange multiplier"
#
# We now want to use Newton's Method to find x* such that
# ∇_x φ(x) = 0
#
# Recall that Newton's method for y(x) = 0 is x ← x - y(x)/y'(x)
#
# Thus, we have: x ← x - [∇^2φ(x)]^-1 ∇φ(x)
#
# We now ask: What is ∇^2φ(x)? (The nxn Hessian)
# ∇^2φ(x) = ∇^2f(x) + ((ρ c(x) + λ) ∇^2c(x) + ρ ∇c(x) * ∇c(x))
# Where the outer product is equivalently ∇c(x) * ∇c(x)'
#
# -------------------------
# Now we take the specific case of the QP
#
# minimize_x (1/2) xT Q x + cT x
# subject to Ax ≤ B
#
# where (all are Real)
# Q is an nxn Symmetric Matrix (that is positive definite for convex)
# x is an nx1 vector
# c is an nx1 vector
# A is an mxn Matrix
# b is an mx1 vector
#
# Then φ(x) = f(x) + (ρ/2) c(x)'c(x) + λ c(x) becomes
# φ(x) = [(1/2) xT Q x + cT x] + (ρ/2) (Ax - b)'(Ax -b) + λ (Ax - b)
#
# ∇φ(x) = ∇f(x) + (ρ c(x) + λ) ∇c(x) becomes
# [Qx + c] + [ρ (Ax - b) + λ] * A
#
# ∇^2φ(x) = ∇^2f(x) + ((ρ c(x) + λ) ∇^2c(x) + ρ ∇c(x) * ∇c(x)) becomes
# Q + ρ A * A
#
# Therefore, Newton's x ← x - [∇^2φ(x)]^-1 ∇φ(x) becomes
# x ← x - inv(Q + ρ A * A) * ([Qx + c] + [ρ (Ax - b) + λ] * A)

using LinearAlgebra

include("backtrackLineSearch.jl")
include("constraints.jl")


function newtonStep(x0, al::augLagQP_AffineIneq)
    #=
    x ← x - [∇^2φ(x)]^-1 ∇φ(x)
    returns [∇^2φ(x)]^-1 ∇φ(x) and ∇φ(x)
    since [∇^2φ(x)]^-1 ∇φ(x) is the step and ∇φ(x) is the residual
    =#
    phiDDinv = inv(evalHessAl(al, x))
    phiD = evalGradAL(al, x0)
    # Note the negative sign!
    return (-phiDDinv * phiD, phiD)
end

function newtonMethodLineSearch(x0, al::augLagQP_AffineIneq, sp::solverParams,
                                            verbose = false)
    xNewtStates = []
    residNewt = []
    push!(xNewtStates, x0)
    xCurr = x0

    lineSearchObj(x) = evalAL(al, x)
    lineSearchdfdx(x) = evalGradAL(al, x)

    # Now, we run through the iterations
    for i in 1:(sp.maxNewtonSteps)
        # Negative sign addressed above
        (dirNewton, residual) = newtonStep(xCurr, al)

        if verbose
            println("Newton Direction: $dirNewton")
        end

        # Then get the line search recommendation
        x0LS, stepLS = backtrackLineSearch(xCurr, dirNewton,
                        lineSearchObj, lineSearchdfdx, sp.paramA, sp.paramB)

        if verbose
            println("Recommended Line Search Step: $stepLS")
            println("Expected x = $x0LS ?= $(xCurr + stepLS * dirNewton)")
        end

        if norm(xCurr - x0LS, 2) < xtol
            break
        else
            xCurr = x0LS
        end

    end

    return xCurr

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

        (xNewStates, resAtStates) = newtonMethodLineSearch(x0, al, sp, verbose)

        # Take each step in the arrays above and save it to the respective
        # overall arrays
        xStates = [xStates; xNewStates]
        residuals = [residuals; resAtStates]

        # Determine the new lambda and rho
        # λ ← λ + ρ c(x_k*)       - Which is to say update with prior x*
        # ρ ← min(ρ * 10, 10^6)   - Which is to say we bound ρ's growth by 10^6
        xNewest = xNewStates[end]
        APost = getGradC(al.constraints, xNewest)
        al.lambda = al.lambda + al.rho * (APost * xNewest - al.constraints.b)
        al.rho = min(al.rho * sp.penaltyStep, sp.penaltyMax)

        if verbose
            println("New state added")
            println("Lambda Updated: $(al.lambda)")
            println("rho updated: $(al.rho)")
        end

        if norm(xNewest - x0, 2) < sp.xTol
            break
        else
            x0 = xNew
        end
    end

    return xStates
end
