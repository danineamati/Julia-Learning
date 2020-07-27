# Holds the possible objective structs


@doc raw"""
Quadratic Objective/Cost Function

    objectiveQP(Q, p)
----

```math
\frac{1}{2} x^{⊤} Q x + p^{⊤} x
```

`Q` is an nxn Symmetric Matrix (that is positive definite for convex)

`p` is an nx1 vector

See also: [`fObjQP`](@ref), [`dfdxQP`](@ref)
"""
struct objectiveQP
    "Positive Definite Symmetric Matrix"
    Q
    p
end

# Input x as a COLUMN vector (i.e. x = [4; 3])
"""
    fObjQP(qp::objectiveQP, x)

Evaluates a quadratic function at the input `x`. The input `x` should be a
*column* vector (i.e. x = [4; 3])
"""
function fObjQP(qp::objectiveQP, x)
    return (1/2) * x' * (qp.Q) * x + (qp.p)' * x
end

@doc raw"""
    dfdxQP(qp::objectiveQP, x)

Evaluates the derivative of a quadratic function at input `x`.

```math
\frac{d}{dx} (\frac{1}{2} x^{⊤} Q x + p^{⊤} x) = Qx + p
```
"""
function dfdxQP(qp::objectiveQP, x)
    return (qp.Q) * x + qp.p
end
