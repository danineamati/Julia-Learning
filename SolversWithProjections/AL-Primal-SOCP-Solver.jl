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

include("SOCP-Setup-Simple.jl")
include("backtrackLineSearch.jl")
include("constraints.jl")
include("trustRegion.jl")



function newtonStepALPSOCP(y0::SOCP_primals, al::augLagQP_2Cone, delta = 1,
                                gamma = 1.5, epsilon = 0.5, a = 1e-6)
    #=
    y ← y - [H(φ(y))]^-1 ∇φ(y)
    returns -[H(φ(y))]^-1 ∇φ(y) and ∇φ(y)
    since -[H(φ(y))]^-1 ∇φ(y) is the step and ∇φ(y) is the residual
    =#
    hess = evalHessAl(al, y0)
    phiD = evalGradAL(al, y0)

    if isposdef(hess) && (rank(hess) == size(hess, 1))
        # Check that the Matrix is positive definite AND non-singular
        # Due to rounding error the hessian is occasionally determined to
        # be positive definite but still singular.

        phiHinv = inv(hess)

        if false
            print("Gradient of AL: ")
            println(phiD)
        end

        # Note the negative sign
        dk = -phiHinv * phiD
        damping = 0
    else
        damping, dk = findDamping(hess, phiD, delta, gamma, epsilon, a)
    end

    if norm(dk) > delta
        dk = (delta / norm(dk)) * dk
    end

    return dk, phiD, hess, damping

end

function newtonLineSearchALPSOCP(y0::SOCP_primals, al::augLagQP_2Cone,
                                        sp::solverParams, verbose = false)
    yNewtStates = SOCP_primals[]
    residNewt = []
    # push!(yNewtStates, y0)
    yCurr = y0

    trustDelta = 2 #10^-10

    # For printing
    early = false

    xSize = size(y0.x, 1)
    sSize = size(y0.s, 1)
    tSize = size(y0.t, 1)

    lineSearchObj(v) = evalAL(al, primalStruct(v, xSize, sSize, tSize))
    lineSearchdfdx(v) = evalGradAL(al, primalStruct(v, xSize, sSize, tSize))

    # Now, we run through the iterations
    for i in 1:(sp.maxNewtonSteps)

        if verbose
            println("Currently at $yCurr")
        end

        currentObjVal = lineSearchObj(primalVec(yCurr))

        if verbose
            println("ϕ(y) = $currentObjVal")
            println("∇ϕ(y) = $(lineSearchdfdx(primalVec(yCurr)))")
            cCurr = getNormToProjVals(al.constraints, yCurr.x, yCurr.s,
                                        yCurr.t, al.lambda[1])
            println("Constraints: $cCurr")
        end

        # Take a Newton Step
        # Negative sign addressed above
        (dirNewton, residual, hess, damp) = newtonStepALPSOCP(
                                                        yCurr, al, trustDelta)
        push!(residNewt, residual)

        if true
            println("\nNewton Direction: $dirNewton")
            println("Damping of ($damp) and trust size of ($trustDelta)")
            # println("AL: $al")
        end

        # Determine if the trust region is sufficient
        y0New = primalVec(yCurr) + dirNewton
        baseObjVal = lineSearchObj(y0New)
        println("Objective from $currentObjVal → $baseObjVal")
        if baseObjVal ≤ currentObjVal
            # Trust region was a success
            # Step 4 of algorithm 3.1 in the Nocedal et Yuan paper

            if true
                println("Trust Region Success")
            end

            fObjApprox(d) = residual'd + (1/2) * d'hess*d

            rho = (currentObjVal - baseObjVal) / (-fObjApprox(dirNewton))
            roundingErrorTol = UInt8(round(-log10(sp.xTol)) / 2)

            println("Rho = $rho")
            roundedDkNorm = round(norm(dirNewton), digits = roundingErrorTol)
            roundedTrustDelta = round(trustDelta, digits = roundingErrorTol)

            print("||dk|| = $(norm(dirNewton)) → $roundedDkNorm")
            print(" vs Δ = $trustDelta → $roundedTrustDelta")
            println(" at $roundingErrorTol digits")

            if rho ≥ 0.2 && roundedDkNorm < roundedTrustDelta
                # Condition 1: No change
                # trustDelta = trustDelta # Probably not needed
                println("Trust Region Size unchanged. Still at $trustDelta")
            elseif rho < 0.2
                trustDelta = (0.2 * norm(dirNewton) + 0.4 * trustDelta) / 2
                println("Trust Region Size decreased. Now $trustDelta")
            else
                trustDelta = 2 * trustDelta
                println("Trust Region Size increased. Now $trustDelta")
            end


        else
            # Trust region failed.

            if true
                print("Trust Region Failed - ")
                println("Trying Line Search.")
            end

            # Attempt a linesearch
            # Get the line search recommendation
            y0New, stepLS = backtrackLineSearch(primalVec(yCurr), dirNewton,
                            lineSearchObj, lineSearchdfdx, sp.paramA, sp.paramB)

            if true
                println("    Recommended Line Search Step: $stepLS")
                if stepLS < 0.125
                    println("    Very low step. Newton Step was $dirNewton")
                end
            end

            if verbose
                print("Expected x = ")
                println("$y0New ?= $(primalVec(yCurr) + stepLS * dirNewton)\n")
            end

            # Now we update the trust region
            yChange = norm(stepLS * dirNewton)
            trustDelta = (yChange + 0.4 * trustDelta) / 2

        end

        # Save the new state to the output list
        y0New_Struct = primalStruct(y0New, xSize, sSize, tSize)
        push!(yNewtStates, y0New_Struct)

        if true
            println("Added State: $y0New_Struct\n")
        end

        # Break by tolerance and trust region size
        if (norm(primalVec(yCurr) - y0New, 2) < sp.xTol) && damp < 1
            println("Ended from tolerance at $i Newton steps\n")
            early = true
            break
        end

        # Update the current state with the new state
        yCurr = y0New_Struct

    end

    if !early
        println("Ended from max steps in $(sp.maxNewtonSteps) Newton Steps\n")
    end

    if verbose
        println("Ended Newton Method at $yCurr")
        println("ϕ(y) = $(lineSearchObj(primalVec(yCurr)))")
        println("∇ϕ(y) = $(lineSearchdfdx(primalVec(yCurr)))")
        cCurr = getNormToProjVals(al.constraints, yCurr.x, yCurr.s,
                                    yCurr.t, al.lambda[1])
        println("Constraints: $cCurr\n")
    end

    return yNewtStates, residNewt

end

function ALPrimalNewtonSOCPmain(y0::SOCP_primals, al::augLagQP_2Cone,
                                sp::solverParams, verbose = false)

    yStates = SOCP_primals[]
    residuals = []
    push!(yStates, y0)

    for i in 1:(sp.maxOuterIters)

        # Update x at each iteration
        if verbose
            println("\n--------------------------------------")
            println("Next Full Update starting at $y0")
        end

        (yNewStates, resAtStates) = newtonLineSearchALPSOCP(y0, al, sp, verbose)

        # Take each step in the arrays above and save it to the respective
        # overall arrays
        yStates = [yStates; yNewStates]
        residuals = [residuals; resAtStates]

        # Determine the new lambda and rho
        # λ ← λ + ρ c(x_k*)       - Which is to say update with prior x*
        # ρ ← min(ρ * 10, 10^6)   - Which is to say we bound ρ's growth by 10^6
        yNewest = yNewStates[end]
        cCurr = getNormToProjVals(al.constraints, yNewest.x, yNewest.s,
                                    yNewest.t, al.lambda[1])

        # Only turn on if the solver is not returning all of the points.
        # if false
        #     println()
        #     println("yStates = $yStates")
        #     println("yNewest = $yNewest")
        #     println("All residuals = $residuals")
        # end
        if verbose
            println("\n################")
            println("Former Lambda: $(al.lambda)")
        end

        lambdaNew = al.lambda + al.rho * cCurr
        al.lambda = [max(lambdaNew[1], 0); lambdaNew[2:end]]
        al.rho = clamp(al.rho * sp.penaltyStep, 0, sp.penaltyMax)

        if verbose
            println("New state added: (Newest) $yNewest")
            println("cCurr: $(cCurr)")
            println("Lambda Updated: $(al.lambda) vs. $lambdaNew")
            println("rho updated: $(al.rho)")
            println("################\n")
        end

        if norm(primalVec(yNewest) - primalVec(y0), 2) < sp.xTol
            println("Ended early at $i outer steps")
            break
        else
            y0 = yNewest
        end
    end

    return yStates, residuals
end
