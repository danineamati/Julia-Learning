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

include("..\\LearningOptimization\\backtrackLineSearch.jl")

function cPlus(A, x, b)
    # This function is for a constraint of the form Ax ≤ b.
    # When Ax - b ≤ 0, we return 0 (Constraint satisfied). When Ax - b > 0
    # We return Ax - b
    #vec([max(ci, 0) for ci in (A * x - b)])
    return max.((A*x - b), 0.0)
end

function cPlusD(A, x, b)
    # If c(x) > 0, then ∇c_+(x) = ∇c(x) = A
    # Else, ∇c_+(x) = 0
    cX = A * x - b
    ANew = deepcopy(A)
    for row in 1:size(A, 1)
        if cX[row] ≤ 0
            # Constraint passed
            ANew[row, :] = zeros(size(A, 2))
        end
    end

    return ANew
end

function getQPgradPhiAL(x, Q, c, A, b, rho, lambda)
    return (Q * x + c) + A' * (rho * (A * x - b) + lambda)
end

function getQPhessPhiAL(Q, A, rho)
    return Q + rho * A'A
end

function newtonStep(phiDDinv, phiD)
    # x ← x - [∇^2φ(x)]^-1 ∇φ(x)
    # returns [∇^2φ(x)]^-1 ∇φ(x)
    return phiDDinv * phiD
end

function newtonMethodLineSearch(x, fObj, dfdx, Q, c, A, b, rho, lambda,
    xtol = 10^-10, maxIters = 5, paramA = 0.1, paramB = 0.5, verbose = false)
    # Note that the [∇^2φ(x)]^-1 is INDEPENDENT of x. So we solve it once and
    # store it.

    phiDDinv = inv(getQPhessPhiAL(Q, A, rho))

    xCurr = x

    # Now, we run through the iterations
    for i in 1:maxIters
        # compute ∇φ(x)
        phiD = getQPgradPhiAL(xCurr, Q, c, A, b, rho, lambda)

        # Note the negative sign!
        dirNewton = -newtonStep(phiDDinv, phiD)

        println("Newton Direction: $dirNewton")

        # Then get the line search recommendation
        x0LS, stepLS = backtrackLineSearch(xCurr, dirNewton, fObj, dfdx,
                                            paramA, paramB)

        println("Recommended Line Search Step: $stepLS")
        println("Expected x = $x0LS ?= $(xCurr + stepLS * dirNewton)")

        if norm(xCurr - x0LS, 2) < xtol
            break
        else
            xCurr = x0LS
        end

    end

    return xCurr

end

function ALNewtonQPmain(x0, fObj, dfdx, Q, c, A, b, rho, lambda,
    xtol = 10^-10, maxIters = 5, paramA = 0.1, paramB = 0.5, verbose = false)

    xStates = []
    push!(xStates, x0)

    rhoIncrease = 10
    rhoMax = 10^6

    for i in 1:maxIters
        # Update x at each iteration
        xNew = newtonMethodLineSearch(x0, fObj, dfdx, Q, c, A, b, rho, lambda,
                            xtol, maxIters, paramA, paramB, verbose)
        push!(xStates, xNew)

        println("New state added")

        # Determine the new lambda and rho
        # λ ← λ + ρ c(x_k*)       - Which is to say update with prior x*
        # ρ ← min(ρ * 10, 10^6)   - Which is to say we bound ρ's growth by 10^6
        lambda = lambda + rho * (A * xNew - b)

        println("Lambda Updated")

        rho = min(rho * rhoIncrease, rhoMax)

        println("Checking tolerance")

        if norm(xNew - x0, 2) < xtol
            break
        else
            x0 = xNew
        end
    end

    return xStates
end
