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


function findDamping(B, g, delta = 1, gamma = 1.5, epsilon = 0.5,
                        a = 0.1, verbose = false)
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
    damping, dampingMax = dampingInitialization(B, g, delta, epsilon, a)

    ratio = 0.1

    dk = - inv(B + ratio * damping * I) * g

    if verbose
        println("Damping Start = $damping")
        println("dk = $dk")
    end

    counter = 1

    while norm(dk) > delta

        rCho = factorize(B + ratio * damping * I).U

        if verbose
            println("Increasing Damping")
            println("Cholesky Factorization:")
            display(rCho)
            println("dk")
            display(dk)
        end

        qk = rCho' \ dk

        updateNumerator = norm(dk)^2 * (gamma * norm(dk) - delta)
        updateDenominator = norm(qk)^2 * delta

        damping = damping + updateNumerator / updateDenominator

        if verbose
            println("qk = $qk")
            println("New Damping: $damping")
        end

        dk = - inv(B + ratio * damping * I) * g
        counter += 1

        if counter >= 10
            break
        end

        if damping >= dampingMax
            damping = min(damping, dampingMax)
            return damping, - inv(B + ratio * damping * I) * g
        end
    end

    return damping, dk

end
