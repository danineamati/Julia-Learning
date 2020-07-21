# Implements trust region code

using LinearAlgebra

function dampingInitialization(B, g, delta, epsilon, a = 0.5)
    #=
    Finds an appropriate damping parameter to start the search

    Specifically,
    λ ∈ {0, ||B|| + (1 + ϵ) ||g|| / Δ}
    =#

    dampingMax = (norm(B) + ((1 + epsilon) * norm(g) / delta))

    if isposdef(B)
        return 0, dampingMax
    else
        while true
            damping = a * dampingMax

            if isposdef(B + damping * I)
                return damping, dampingMax
            else
                a = (a + 1)/ 2
            end
        end
    end
end


function findDamping(B, g, delta = 1, gamma = 1.5, epsilon = 0.5)
    #=
    Implements a search for the appropriate trust region.

    B           is the approximate or true Hessian matrix.
    g           is the gradient of the function
    delta      is the initial trust region size
    gamma > 1   is a parameter for how aggressive to increase the
                            trust region size
    epsilon > 0 is a parameter which weighes the gradient versus the
                            hessian in determining the damoping ratio size


    return damping (damping parameter)

    This corresponds to algorithm 2.6 in Nocedal et Yuan (1998)
    =#
    damping, dampingMax = dampingInitialization(B, g, delta, epsilon)
    println("Damping Start = $damping")

    dk = -g \ (B + damping * I)
    println("dk = $dk")

    counter = 1

    while norm(dk) > delta
        println("Increasing Damping")
        rCho = factorize(B + damping * I).U
        println("Cholesky Factorization:")
        display(rCho)
        qk = dk' \ rCho'
        println("qk = $qk")

        updateNumerator = norm(dk)^2 * (gamma * norm(dk) - delta)
        updateDenominator = norm(qk)^2 * delta

        damping = damping + updateNumerator / updateDenominator
        println("New Damping: $damping")

        dk = -g \ (B + damping * I)
        counter += 1

        if counter >= 10
            break
        end

        if damping >= dampingMax
            damping = min(damping, dampingMax)
            return damping, -g \ (B + damping * I)
        end
    end

    return damping, dk

end
