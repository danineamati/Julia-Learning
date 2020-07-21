# Implements trust region code

using LinearAlgebra

function dampingInitialization(B, g, delta0, epsilon)
    #=
    Finds an appropriate damping parameter to start the search

    Specifically,
    λ ∈ {0, ||B|| + (1 + ϵ) ||g|| / Δ}
    =#
    if isposdef(B)
        return 0
    else
        return 0.5 * (norm(B) + ((1 + epsilon) * norm(g) / delta0))
    end
end


function findTrustRegionSize(B, g, delta0 = 1, gamma = 1.5, epsilon = 0.5)
    #=
    Implements a search for the appropriate trust region.

    B           is the approximate or true Hessian matrix.
    g           is the gradient of the function
    delta0      is the initial trust region size
    gamma > 1   is a parameter for how aggressive to increase the
                            trust region size
    epsilon > 0 is a parameter which weighes the gradient versus the
                            hessian in determining the damoping ratio size


    return lambda (damping parameter) and delta (trust region size)

    This corresponds to algorithm 2.6 in Nocedal et Yuan (1998)
    =#
    gamma = dampingInitialization(B, g, delta0, epsilon)

end
