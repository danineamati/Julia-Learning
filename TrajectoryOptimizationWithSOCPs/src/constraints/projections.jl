
using LinearAlgebra

# Covers how to consider a square violation
function sqrDistEuclid(pt, projFunc)
    #=
    Choose a given projection method and determine the squared distance
    at this point.
    =#
    return norm(pt - projFunc(pt), 2)^2
end



# Covers Projections

# First, Lines from vectors at the origin
function projLine(pt, vec)
    #=
    This function projects a point pt onto a line described by vec

    proj_v(p) = v * (p'v)/(v'v)
    =#
    return vec * (pt'vec) / (vec'vec)
end

function orthoProjLine(pt, vec)
    #=
    This function projects a point pt onto the orthogonal of a line described
    by vec

    proj_v(p) = v * (p'v)/(v'v)
    proj_v⟂(p) = pt - v * (p'v)/(v'v)
    =#
    return pt - projLine(pt, vec)
end

# Second, Positive Orthant
function projPosOrth(pt)
    #=
    This function is for a constraint of the form "τ ≥ 0."
    When τ_i ≥ 0, the variable τ is in the positive orthant, thus, the
    projection is simply τ_i.
    When τ_i < 0, the variable τ is not in the positive orthant, thus, the
    projection is distance to the positive orthant
    =#

    return max.(pt, 0)
end

# Third, Affine Case
function projAffineEq(a, b, x)
    #=
    Projection for Affine Equality Case (a'x = b)

    We can show that the projection is given by
    proj(x) = x + (b - a'x) * a / a'a
            = x + a * [b] / a'a - a * (a'x) / a'a
    =#
    t1 = a * b / a'a
    t2 = a * (a'x) / a'a
    return x + t1 - t2
end


function projAffineIneq(a, b, x)
    #=
    Projection for Affine Inequality Case (a'x ≤ b)

    We can show that the projection is given by
    projAffineEq(x) for a'x > b
    x               for a'x ≤ b
    =#
    if a'x > b
        return projAffineEq(a, b, x)
    end
    return x
end

function projSecondOrderCone(v, s, filled = true, p = 2)
    #=
    Projection for Second-Order Cone (AKA quadratic cone or the Lorentz cone)

    The second-order cone is C = {(x, t) ∈ R^n+1 | ||x||2 ≤ t}. Using the
    2-norm. Projection onto it is given by

    proj(v, s) =
    0                                   for ||v|| ≤ -s  (Below the tip)
    (v, s)                              for ||v|| ≤ s   (In the cone)
    (1/2)(1 + s/||v||)(v, ||v||)        for ||v|| ≥ |s| (Onto the cone)

    Note that (|s| = absolute value of s)
    =#

    if norm(v, p) ≤ -s
        # println("Below Tip")
        return zeros(size([v; s]))
    elseif norm(v, p) ≤ s
        # println("Inside already")
        if filled
            return [v; s]
        end
        if !filled
            return (1/2) * (1 + s / norm(v, p)) * [v; norm(v, p)]
        end
    elseif norm(v, p) ≥ abs(s)
        # println("Outside")
        return (1/2) * (1 + s / norm(v, p)) * [v; norm(v, p)]
    end

    println("Second Order Cone Conditions ERROR")
    return -1

end
