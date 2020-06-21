# Primal-Dual Interior Point Newton Method
#
# This is specifically for QPs and Interior Point
#
#
# minimize_x f(x)
# subject to Ax ≦ b
#
# We rewrite this as f(x) - λT g(x)
# Where g(x) are the constraints such that c(x) ≧ 0 and λ ≧ 0
# The first-order condition of the KKT is
# ∇f(x) - λT ∇g(x) = 0
#
# Now we want to use a Primal-Dual trick for a Log-Barrier
# (i.e. Interior Point). We have
#
# minimize_x f(x) - μ log(g(x))
# The first-order condition of the KKT is
# ∇f(x) - μ ∇g(x) / g(x) = 0
#
# So, λT = μ / g(x)
#
# -------------------------
#
# Now, we want to use Newton's Method.
#
# We make a vector h = [x λ]T and r = r(x, λ) and we update h as
# h ← h + ∇r^{-1} r
# Where
# r = (∇f(x) + λT ∇g(x))
#     ( -diag(λ) g(x) - μ 1)
#
# So ∇r = [A B; C D]
# Where:
# A = ∇^2 f(x) + λT ∇^2 g(x)
# B = ∇g(x)^T
# C = - diag(λ) ∇g(x)
# D = -diag(g(x))
#
# --------------------------
#
# We can simplify with the QP assumption
#
# minimize_x (1/2) xT Q x + cT x
# subject to Ax ⩽ B
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
# # r = (  ∇f(x) + λT ∇g(x) )  = (        Qx + c + λT A        )
#       (-diag(λ) g(x) - μ 1)  = (-diag(λ) Ax - diag(λ) b - μ 1)
#
# So ∇r = [A B; C D]
# Where:
# A = ∇^2 f(x) + λT ∇^2 g(x) = Q
# B = ∇g(x)^T                = A^T
# C = - diag(λ) ∇g(x)        = -diag(λ) A
# D = -diag(g(x))            = -diag(Ax - b)
#
# ----------------------------
# Therefore we need two parts, the outer part that updates μ and
# the inner part that runs the Newton Update
#
# This script has the inner part first, then the outer part

# ------------------
# Schur complements for Matrix Inverse
# -----------------

using LinearAlgebra

include("..\\LearningOptimization\\backtrackLineSearch.jl")
include("QP-Setup.jl")



function schurComplement(A, B, C, D)
    # Takes Four Matrices A, B, C, D
    # where the original Matrix is:
    # [ A  B ]
    # [ C  D ]
    # The Schur Complement is
    # inv(A - B inv(D) C)

    return inv(A - B * inv(D) * C)

end

function invWithSchurComplement(A, B, C, D)
    # Uses the schur complement to produce the inverse

    # First get the schurComplement
    isc = schurComplement(A, B, C, D)
    iD = inv(D)

    topLeft = isc
    topRight = - isc * B * iD
    botLeft = - iD * C * isc
    botRight = iD + iD * C * isc * B * iD

    return [topLeft topRight; botLeft botRight]
end

function getQPrVec(Q, c, A, b, x, lambda, mu)
    # # r = (  ∇f(x) + λT ∇g(x) )  = (        Qx + c + λT A        )
    #       (-diag(λ) g(x) - μ 1)  = (-diag(λ) Ax - diag(λ) b - μ 1)
    @assert size(A, 1) == size(lambda, 1)
    @assert size(x, 1) == size(c, 1)
    @assert size(Q, 2) == size(x, 1)

    r1 = Q * x + c + A'lambda
    r2 = - Diagonal(lambda) * A * x - Diagonal(lambda) * b - mu * ones(size(A, 1))

    return vcat(r1, r2)
end

function getQPGradrVec(Q, A, b, x, lambda)
    # So ∇r = [A B; C D]
    # Where:
    # A = ∇^2 f(x) + λT ∇^2 g(x) = Q
    # B = ∇g(x)^T                = A^T
    # C = - diag(λ) ∇g(x)        = -diag(λ) A
    # D = -diag(g(x))            = -diag(Ax - b)
    Ar = Q
    Br = A'
    Cr = - Diagonal(lambda) * A
    Dr = - Diagonal(A * x - b)

    return (Ar, Br, Cr, Dr)
end

function newtonStep(gRA, gRB, gRC, gRD, rV)
    # h ← h + ∇r^{-1} r

    gRInv = invWithSchurComplement(gRA, gRB, gRC, gRD)
    return gRInv * rV
end






# Call functions
println()
println("Testing Newton Step on test matrix")
matA = [4 5 9; 3 2 1; 0 9 10]
matB = [2 4 5; 8 6 7; 1 4 2]
matC = [8 5 7; 3 1 6; 4 9 2]
matD = [4 5 2; 8 6 4; 1 2 3]

rVec = [3; 4; 5; 9; 8; 7]

nextStep = newtonStep(matA, matB, matC, matD, rVec)
display(nextStep)

println()
println("Setting Up the QP")
QMat, cVec, AMat, bVec, x0 = QPSetup()

fObj(x) = (1/2) * x'QMat*x + cVec'x
dfdx(x) = QMat * x + cVec

println("Objective: f(x) = (1/2) x'Qx + c'x")
println("Q = $QMat")
println("c = $cVec")
println("Constraints: Ax ≦ b")
println("A = $AMat")
println("b = $bVec")
println("Initial Starting Point: $x0")

println()
println("Testing r Vec from QP-Setup")
lambda = [1; 1; 1]
mu = 1
rVec = getQPrVec(QMat, cVec, AMat, bVec, x0, lambda, mu)
grA, grB, grC, grD = getQPGradrVec(QMat, AMat, bVec, x0, lambda)
println("r vec = $rVec")
println("∇r =")
display([grA grB; grC grD])

dirNewton = newtonStep(grA, grB, grC, grD, rVec)
println()
println("First Newton Step: ")
display(dirNewton)

# stepAfterLineSearch = backtrackLineSearch()

println("Completed")
