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


function findDamping(B, g, delta = 1, gamma = 1.5, epsilon = 0.5,
                        condNumMax = 1e7, a = 1e-6, verbose = false)
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
    damping, dampingMax, rCho = dampingInitialization(B, g,
                                                            delta, epsilon, a)

    # Use the Cholesky decomposition to more easily solve the problem.
    dk = - rCho.U \ (rCho.U' \ g)

    if verbose
        println("Damping Start = $damping and dampingMax = $dampingMax")
        println("dk = $dk")
    end

    counter = 1

    while (norm(dk) > delta) && (cond(rCho.U'rCho.U) > condNumMax)

        if verbose
            println("Increasing Damping")
            println("Cholesky Factorization:")
            display(rCho)
            println("dk")
            display(dk)
        end

        qk = rCho.U' \ dk

        updateNumerator = norm(dk)^2 * (gamma * norm(dk) - delta)
        updateDenominator = norm(qk)^2 * delta

        damping = min(damping + updateNumerator / updateDenominator, dampingMax)

        if verbose
            println("qk = $qk")
            println("New Damping: $damping")
        end

        rCho = cholesky(B + damping * I, check = false)

        if issuccess(rCho)
            dk = - rCho.U \ (rCho.U' \ g)
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

    println("Condition Number: $(cond(rCho.L * rCho.U))")

    return damping, dk, rCho

end
