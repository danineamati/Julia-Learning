# Primal-Dual Augmented Lagrangian Newton Method
#
# This is specifically for QPs and Augmented Lagrangian
#
#
# minimize_x f(x)
# subject to Ax ≤ b
#
# We rewrite this as f(x) + λT g(x)
# Where g(x) are the constraints such that c(x) ≦ 0 and λ ≧ 0
# The first-order condition of the KKT is
# ∇f(x) + λT ∇g(x) = 0
#
# Now we want to use a Primal-Dual trick for an Augmented Lagrangian
# (i.e. squared penalty + lagrange multiplier). We have
#
# minimize_x f(x) + (ρ/2) Σ ||g_i(x)||_2^2 + Σ ν_i g_i(x)
# The first-order condition of the KKT is
# ∇f(x) + ρ Σ g_i(x) * ∇g_i(x) + Σ ν_i ∇g_i(x) = 0
#∇f(x) + Σ (ρ g_i(x) + ν_i) * ∇g_i(x) + Σ∇g_i(x) = 0
# So, ρ g_i(x) + ν_i = λ_i
# Or, equivalently, g(x) + (1/ρ) (ν - λ) = 0
#
# -------------------------
#
# Now, we want to use Newton's Method.
#
# We make a vector h = [x λ]T and r = r(x, λ) and we update h as
# h ← h + ∇r^{-1} r
# Where
# r = (  ∇f(x) + λT ∇g(x)  )
#     ( g(x) + (1/ρ)(ν - λ))
#
# So ∇r = [A B; C D]
# Where:
# A = ∇^2 f(x) + λT ∇^2 g(x)
# B = ∇g(x)^T
# C = ∇g(x)
# D = -(1/ρ)
#
# --------------------------
#
# We can simplify with the QP assumption
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
#
# So,
# ∇f(x) = Qx + c
# ∇^2f(x) = Q
#
# ∇g(x) = ∇(Ax - b) = A
# ∇^2g(x) = 0
#
# Thus,
# r = (  ∇f(x) + λT ∇g(x)  ) = (    Qx + c + λT * A   )
#     ( g(x) + (1/ρ)(ν - λ)) = ( Ax - b + (1/ρ)(ν - λ))
#
# So ∇r = [A B; C D]
# Where:
# A = ∇^2 f(x) + λT ∇^2 g(x) = Q
# B = ∇g(x)^T                = A^T
# C = ∇g(x)                  = A
# D = -(1/μ) * I             = -(1/μ) * I
#
# ----------------------------
# Therefore we need two parts, the outer part that updates μ and
# the inner part that runs the Newton Update
#
# This script has the inner part first, then the outer part



using LinearAlgebra

include("backtrackLineSearch.jl")
include("constraints.jl")
include("schurInverse.jl")


function getQPrVecAL(Q, c, A, b, x, lambda, rho, nu)
    #=
    r = (     ∇f(x) + λT ∇g(x)     ) = (    Qx + c + λT * A   )
        ( g_i(x) + (1/μ)(ρ_i - λ_i)) = ( Ax - b + (1/μ)(ρ - λ))

    Notice that r1 is not the gradient of the augmented lagrangian.
    =#

    APost = cPlusD(A, x, b)

    r1 = (Q * x + c) + APost'lambda
    r2 = cPlus(A, x, b) + (1/rho) * (nu - lambda)

    return vcat(r1, r2)
end

function getQPGradrVecAL(Q, A, b, x, rho)
    # So ∇r = [A B; C D]
    # Where:
    # A = ∇^2 f(x) + λT ∇^2 g(x) = Q
    # B = ∇g(x)^T                = A^T
    # C = ∇g(x)                  = A
    # D = -(1/μ)                 = -(1/μ)

    # Note that the APost matrix is used to account for inequality constraints
    APost = cPlusD(A, x, b)

    Ar = Q
    Br = APost'
    Cr = APost
    Dr = - (1/rho) * Diagonal(ones(size(A,1)))

    return (Ar, Br, Cr, Dr)
end

function newtonStep(gRA, gRB, gRC, gRD, rV)
    # h ← h - ∇r^{-1} r
    # Here we return ∇r^{-1} r

    gRInv = invWithSchurComplement(gRA, gRB, gRC, gRD)
    return gRInv * rV
end

function newtonAndLineSearch(Q, c, A, b, hV, rho, nu, phiObj, dphidx,
                                paramA = 0.1, paramB = 0.5, verbose = false)
    # Current state
    xSize = size(Q, 1)
    if verbose
        println("ρ = $rho")
        # @assert rho > 0
    end

    xCurr = hV[1:xSize]
    lamCurr = hV[xSize + 1:end]

    # Get current r Vector and gradient of r Vector
    rV = getQPrVecAL(Q, c, A, b, xCurr, lamCurr, rho, nu)
    gRA, gRB, gRC, gRD = getQPGradrVecAL(Q, A, b, xCurr, rho)

    # First get the newton step
    # NOTICE the negative sign
    dirNewton = -newtonStep(gRA, gRB, gRC, gRD, rV)

    xMax = xCurr + dirNewton[1:xSize]
    if verbose
        println("Direction: $dirNewton")
        println("Max x = $xMax at φ(x) = $(phiObj(xMax))")
    end
    # Then get the line search recommendation
    # Note that phiObj is the full Augmented Lagrangian
    x0LS, stepLS = backtrackLineSearch(xCurr, dirNewton[1:xSize],
                                    phiObj, dphidx, paramA, paramB, false)
    if verbose
        println("Line Search step = $stepLS")
    end

    # Update the rVector
    # Negative Sign Accounted Above
    x0New = xCurr + stepLS * dirNewton[1:xSize]
    lambdaNew = lamCurr + stepLS * dirNewton[xSize + 1:end]
    hVNew = vcat(x0New, lambdaNew)

    # ONLY reason for the separation is for printing.

    if verbose
        println("x0New = $x0New")
        println("LambdaNew = $lambdaNew")
    end

    return hVNew

end

function pdALNewtonQPmain(Q, c, A, b, x0, lambda, rho, nu, fObj, dfdx,
                    maxIters = 10, paramA = 0.1, paramB = 0.5, verbose = false)
    hCurr = vcat(x0, lambda)
    hStates = []
    push!(hStates, hCurr)

    rho = 1
    rhoIncrease = 10

    for i in 1:maxIters
        # Update rVec at each iteration
        # φ(x) = f(x) + (ρ/2) c(x)'c(x) + λ c(x)
        phi(x) = fObj(x) + (rho / 2) * cPlus(A, x, b)'cPlus(A, x, b) +
                        nu' * cPlus(A, x, b)
        dPhidx(x) = getQPgradPhiAL(x, Q, c, A, b, rho, nu)

        if verbose
            xStart = hCurr[1:size(x0, 1)]
            println("---------")
            println("Starting Outer at: $xStart with value φ(x) = $(phi(xStart))")
        end

        hCurr = newtonAndLineSearch(Q, c, A, b, hCurr, rho, nu,
                                        phi, dPhidx, paramA, paramB, verbose)
        push!(hStates, hCurr)

        # Determine the new lambda and rho
        # ν ← ν + ρ c(x_k*)       - Which is to say update with prior x*
        # ρ ← min(ρ * 10, 10^6)   - Which is to say we bound ρ's growth by 10^6
        xNew = hCurr[1:size(x0, 1)]

        if verbose
            println("New state added: $xNew with value φ(x) = $(phi(xNew))")
        end

        nu = nu + rho * (A * xNew - b)
        rho = rho * rhoIncrease

        if verbose
            println("ν Updated: $nu")
            println("ρ updated: $rho")
        end
    end

    return hStates
end
