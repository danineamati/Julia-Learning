# Implements trust region code

using LinearAlgebra, SparseArrays


@doc raw"""
    dampingInitialization(B, g, delta, epsilon, a = 0.5)

Finds an appropriate damping parameter to start the search.

Specifically,
```math
λ ∈ [0, ||B|| + (1 + ϵ) \frac{||g||}{Δ}]
```

## Arguments
- `B`: The (approximate or true) Hessian Matrix
- `g`: The (approximate or true) gradient vector
- `epsilon`: Parameter (> 0) weighes the gradient versus the hessian in
determining the damping ratio size
- `a`: Parameter (> 0) How quickly to damp the Hessian

See also: [`findDamping`](@ref)
"""
function dampingInitialization(B, g, delta, epsilon, a = 0.5)
    dampingMax = (norm(B) + ((1 + epsilon) * norm(g) / delta))

    if isposdef(B)
        damping = 1e-6
        Bdamped = B + damping * I
        cholBdamp = cholesky(Bdamped, check = false)

        if issuccess(cholBdamp)
            return damping, dampingMax, cholBdamp
        end
    end

    while true
        damping = a * dampingMax

        Bdamped = B + damping * I

        # Get the Cholesky decomposition without throwing an exception
        cholBdamp = cholesky(Bdamped, check = false)

        # Checks positive definite and non-singular
        if issuccess(cholBdamp)
            return damping, dampingMax, cholBdamp
        else
            # If the it fails, increase damping factor
            a = (a + 1)/ 2
        end
    end
end

"""
    findDamping(B, g, delta = 1, gamma = 1.5, epsilon = 0.5,
                      condNumMax = 1e7, a = 1e-6, verbose = false)

Implements a search for the appropriate trust region.

## Arguments:
- `B`: The (approximate or true) Hessian Matrix
- `g`: The (approximate or true) gradient vector
- `delta`: is the initial trust region size
- `condNumMax`: The max condition number on the damped Hessian

for `gamma`, `epsilon`, and `a`, see [`dampingInitialization`](@ref)

returns the damping factor `damping`, the newton step `dk`, and the damped
hessian represented as a Cholesky Decomposition `rCho`

This corresponds to algorithm 2.6 in Nocedal et Yuan (1998)

"""
function findDamping(B, g, delta = 1, gamma = 1.5, epsilon = 0.5,
                        condNumMax = 1e7, a = 1e-6, verbose = false)

    # Initialize the damping factor and find the initial damped hessian
    # represented as a Cholesky Decomposition (rCho)
    damping, dampingMax, rCho = dampingInitialization(B, g, delta, epsilon, a)

    # Use the Cholesky decomposition to more easily solve the problem.
    LCho = sparse(rCho.L)
    dk = - LCho' \ (LCho \ g)

    if verbose
        println("Damping Start = $damping and dampingMax = $dampingMax")
        println("dk = $dk")
    end

    counter = 1

    # Have to use Array to convert briefly to a dense matrix in order
    # to determine the condition number.
    while (norm(dk) > delta) && (cond(Array(LCho * LCho')) > condNumMax)

        if verbose
            println("Increasing Damping")
            println("Cholesky Factorization:")
            display(rCho)
            println("dk")
            display(dk)
        end

        qk = LCho \ dk

        updateNumerator = norm(dk)^2 * (gamma * norm(dk) - delta)
        updateDenominator = norm(qk)^2 * delta

        damping = min(damping + updateNumerator / updateDenominator, dampingMax)

        if verbose
            println("qk = $qk")
            println("New Damping: $damping")
        end

        rCho = cholesky(B + damping * I, check = false)

        if issuccess(rCho)
            LCho = sparse(rCho.L)
            dk = - LCho' \ (LCho \ g)
            counter += 1

            if damping == dampingMax
                break
            end
        else
            # Need to really increase the damping factor
            damping *= 10
        end


        if counter >= 10
            break
        end
    end

    println("Condition Number: $(cond(Array(LCho * LCho')))")

    return damping, dk, rCho

end
