# Contains the rocket set up and structs

#=
Now we want to set up the rocket dynamics. We have a linear system:
xk+1 = Axk + Buk + G

where xk = [sk; vk] (the position and velocity, respectively). If sk is n×1,
A is 2n×2n, B is 2n×n, and G is a 2n×1 vector.

Thus, we get the system:
[A B -I] * [xk; uk; xk+1] = G
which we can stack as

[A B -I 0 0 0 0 0 0]
[0 0 A B -I 0 0 0 0]
[0 0 0 0 A B -I 0 0]
[0 0 0 0 0 0 A B -I]
=#

using SparseArrays, LinearAlgebra

"""
    rocket_simple

Barebones "spherical" rocket. This is to say, this struct only holds the mass
and the isp (specific impulse) of the rocket.
"""
struct rocket_simple
    mass
    isp
    grav
end


@doc raw"""
    rocketDynamics(r::rocket_simple, nDim::Int64)

We have a linear dynamics system
```math
x_{k+1} = A x_k + B u_k + C
```
We compute and return A, B, and C for a [`simple rocket`](@ref rocket_simple).

For a simple rocket, we have

```math
A = [0 \ I; 0 \ 0] \quad B = [0; \frac{1}{m} I] \quad C = [0; -g]
```

"""
function rocketDynamics(r::rocket_simple, nDim::Int64)
    sp0 = spzeros(nDim, nDim)
    aMat = [sp0 I; sp0 sp0]
    bMat = [sp0; (1/r.mass) * I]
    cVec = [spzeros(nDim, 1); r.grav]
    return aMat, bMat, cVec
end

@doc raw"""
We want to take the per discretization dynamics and write it as a full matrix.

Specifically, we get the system:
[A B -I] * [xk; uk; xk+1] = G
which we can stack as

```math
[A \ B \ -I \ 0 \ 0 \ 0 \ 0 \ 0 \ 0]
```

```math
[0 \ 0 \ A \ B \ -I \ 0 \ 0 \ 0 \ 0]
```

```math
[0 \ 0 \ 0 \ 0 \ A \ B \ -I \ 0 \ 0]
```

```math
[0 \ 0 \ 0 \ 0 \ 0 \ 0 \ A \ B \ -I]
```

We return the above matrix
"""
function rocketDynamicsFull(r::rocket_simple, nDim::Int64, NSteps::Int64)
    # First calculate the A and B matrices. The C vector is not needed in this
    # case.
    Ak, Bk, Ck = rocketDynamics(r::rocket_simple, nDim::Int64)
    ABI_unit = [Ak Bk -I]
    colLen = size(ABI_unit, 2)

    # At each step we go down 2 * n rows and move right 3 * n columns.
    sizeCols = 3 * NSteps * nDim + 2 * nDim
    sizeRows = 2 * nDim * NSteps

    AFull = spzeros(sizeRows, sizeCols)

    rStart = 1
    cStart = 1
    for k in 1:NSteps
        rEnd = rStart + (2 * nDim - 1)
        cEnd = cStart + colLen - 1

        println("Accessing $rStart:$rEnd and $cStart:$cEnd")

        AFull[rStart:rEnd, cStart:cEnd] = ABI_unit

        rStart += 2 * nDim
        cStart += 3 * nDim
    end
    return AFull
end
