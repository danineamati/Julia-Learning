# First, we are going to do a normal augmented lagrangian option
# "normal" in the sense of not Primal-Dual
#
# Let's start with the base optimization problem
#
# minimize_x f(x)
# subject to c(x) ≤ 0
#
# The corresponding augmented lagrangian is
# minimize_x φ(x) = f(x) + (ρ/2) ||c(x)||_2^2 + λ c(x)
#
# which we can write:
# f(x) + (ρ/2) c(x)'c(x) + λ c(x)
#
# Note that we solve this problem sequentially. At each iteration, we hold
# ρ and λ fixed and solve minimize_x φ(x).
# At the end of the iteration,
#        ρ ← min(ρ * 10, 10^6)   - Which is to say we bound ρ's growth by 10^6
#        λ ← λ + ρ c(x_k*)       - Which is to say update with prior x*
#
# The first order condition for minimize_x φ(x) is
# ∇_x φ(x) = 0 = ∇f(x) + (ρ c(x) + λ) ∇c(x)
# Thus, (ρ c(x) + λ) acts as a "modified lagrange multiplier"
#
# We now want to use Newton's Method to find x* such that
# ∇_x φ(x) = 0
#
# Recall that Newton's method for y(x) = 0 is x ← x - y(x)/y'(x)
#
# Thus, we have: x ← x - [∇^2φ(x)]^-1 ∇φ(x)
#
# We now ask: What is ∇^2φ(x)? (The nxn Hessian)
# ∇^2φ(x) = ∇^2f(x) + ((ρ c(x) + λ) ∇^2c(x) + ρ ∇c(x) ⊗ ∇c(x))
# Where the outer product is equivalently ∇c(x) * ∇c(x)'
#
# -------------------------
# Now we take the specific case of the QP
#
# minimize_x (1/2) xT Q x + cT x
# subject to Ax ≤ B
#
# where (all are Real)
# Q is an nxn Symmetric Matrix (that is positive definite for convex)
# x is an nx1 vector
# c is an nx1 vector
# A is an mxn Matrix
# b is an mx1 vector
#
# Then φ(x) = f(x) + (ρ/2) c(x)'c(x) + λ c(x) becomes
# φ(x) = [(1/2) xT Q x + cT x] + (ρ/2) (Ax - b)'(Ax -b) + λ (Ax - b)
#
# ∇φ(x) = ∇f(x) + (ρ c(x) + λ) ∇c(x) becomes
# [Qx + c] + [ρ (Ax - b) + λ] * A
#
# ∇^2φ(x) = ∇^2f(x) + ((ρ c(x) + λ) ∇^2c(x) + ρ ∇c(x) * ∇c(x)) becomes
# Q + ρ A * A
#
# Therefore, Newton's x ← x - [∇^2φ(x)]^-1 ∇φ(x) becomes
# x ← x - inv(Q + ρ A * A) * ([Qx + c] + [ρ (Ax - b) + λ] * A)
