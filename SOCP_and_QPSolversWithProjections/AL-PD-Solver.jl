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

include("QP-Setup-Simple.jl")
include("backtrackLineSearch.jl")
include("constraints.jl")
include("schurInverse.jl")


function getQP_PDVecAL(x0, nu0, al::augLagQP_AffineIneq)
    #=
    x is the primal variable
    ν is the  dual  variable

    r = (     ∇f(x) + νT ∇g(x)     ) = (    Qx + c + νT * A   )
        ( g_i(x) + (1/ρ)(λ_i - ν_i)) = ( Ax - b + (1/ρ)(λ - ν))

    Notice that r1 is NOT the gradient of the augmented lagrangian.
    =#

    APost = getGradC(al.constraints, x0)
    gradf = dfdxQP(al.obj, x0)
    cCurr = getNormToProjVals(al.constraints, x0)

    r1 = gradf + APost' * nu0
    r2 = cCurr + (1/(al.rho)) * (al.lambda - nu0)

    return vcat(r1, r2)
end

function getQPGrad_PDVecAL(x0, al::augLagQP_AffineIneq)
    # So ∇r = [A B; C D]
    # Where:
    # A = ∇^2 f(x) + λT ∇^2 g(x) = Q
    # B = ∇g(x)^T                = A^T
    # C = ∇g(x)                  = A
    # D = -(1/ρ)                 = -(1/ρ)

    # Note that the APost matrix is used to account for inequality constraints
    APost = getGradC(al.constraints, x0)

    Ar = al.obj.Q
    Br = APost'
    Cr = APost
    Dr = - (1/(al.rho)) * Diagonal(ones(size(APost,1)))

    return (Ar, Br, Cr, Dr)
end

function newtonStepPDAL(x0, nu0, al::augLagQP_AffineIneq)
    #=
    x is the primal variable
    ν is the  dual  variable
    h is [x; ν]

    h ← h - ∇r^{-1} r
    returns -∇r^{-1} r and r[1:size(x0, 1)]
    since -∇r^{-1} r is the step and r[1:size(x0, 1)] is the primal residual
    =#
    rV = getQP_PDVecAL(x0, nu0, al)
    grA, grB, grC, grD = getQPGrad_PDVecAL(x0, al)

    grInv = invWithSchurComplement(grA, grB, grC, grD)
    return -grInv * rV, rV[1:size(x0, 1)]
end

function newtonAndLineSearchPDAL(h0, al::augLagQP_AffineIneq, sp::solverParams,
                                            verbose = false)
    # (Q, c, A, b, hV, rho, nu, phiObj, dphidx,
    #                             paramA = 0.1, paramB = 0.5, verbose = false)
    #=
    x is the primal variable
    ν is the  dual  variable
    h is [x; ν]
    =#
    # Current state
    xSize = size(al.obj.Q, 1)

    xCurr = h0[1:xSize]
    nuCurr = h0[xSize + 1:end]

    # Save results
    hNewtStates = []
    residNewt = []

    # For printing
    early = false

    lineSearchObj(x) = evalAL(al, x)
    lineSearchdfdx(x) = evalGradAL(al, x)

    # Now, we run through the iterations
    for i in 1:(sp.maxNewtonSteps)
        # First get the newton step
        # note that the negative sign is already addressed above
        (dirNewton, residual) = newtonStepPDAL(xCurr, nuCurr, al)
        push!(residNewt, residual)

        if verbose
            println("Newton Direction: $dirNewton at $xCurr")
        end

        # Then get the line search recommendation
        x0LS, stepLS = backtrackLineSearch(xCurr, dirNewton[1:xSize],
                        lineSearchObj, lineSearchdfdx, sp.paramA, sp.paramB)

        if verbose
            println("Recommended Line Search Step: $stepLS")
            println("Expected x = $x0LS ?= $(xCurr + stepLS * dirNewton[1:xSize])")
        end

        x0New = xCurr + stepLS * dirNewton[1:xSize]
        nuNew = nuCurr + stepLS * dirNewton[xSize + 1:end]

        h0New = [x0New; nuNew]

        push!(hNewtStates, h0New)

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

    return hNewtStates, residNewt

end

function ALPDNewtonQPmain(x0, al::augLagQP_AffineIneq, sp::solverParams,
                                            verbose = false)
    n0 = zeros(size(al.constraints.A, 1))

    xSize = size(x0, 1)

    hCurr = vcat(x0, n0)
    hStates = []
    residuals = []
    push!(hStates, hCurr)

    for i in 1:(sp.maxOuterIters)

        if verbose
            xStart = hCurr[1:xSize]
            println("---------")
            println("Starting Outer at: $xStart")
        end

        (hNewStates, resAtStates) = newtonAndLineSearchPDAL(hCurr, al, sp, verbose)

        # Take each step in the arrays above and save it to the respective
        # overall arrays
        hStates = [hStates; hNewStates]
        residuals = [residuals; resAtStates]

        # Determine the new lambda and rho
        # λ ← λ + ρ c(x_k*)       - Which is to say update with prior x*
        # ρ ← min(ρ * 10, 10^6)   - Which is to say we bound ρ's growth by 10^6
        xNewest = hNewStates[end][1:xSize]
        APost = getGradC(al.constraints, xNewest)

        if verbose
            println()
            println("APost = $APost")
            # println("xStates = $hStates")
            println("xNewest = $xNewest")
            println("New residuals = $resAtStates")
        end
        lambdaNew = al.lambda + al.rho * (APost * xNewest - al.constraints.b)
        al.lambda = max.(lambdaNew, 0)
        al.rho = min(al.rho * sp.penaltyStep, sp.penaltyMax)
        # al.rho = al.rho * sp.penaltyStep

        if verbose
            println("New state added")
            println("Lambda Updated: $(al.lambda)")
            println("rho updated: $(al.rho)")
        end

        if norm(xNewest - x0, 2) < sp.xTol
            println("Ended early at $i outer steps")
            break
        else
            hCurr = hNewStates[end]
        end
    end

    return hStates, residuals
end
