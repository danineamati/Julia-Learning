using LinearAlgebra

# ------------------------
# Basic Linear Algebra
# -----------------------

matA = [4 5 9; 3 2 1; 0 9 10]
println(matA)

println("Transpose: ")
display(matA')

println("Diagonal: ")
display(Diagonal(matA))

println("Inverse: ")
display(inv(matA))

println("Inverse of Diagonal: ")
display(inv(Diagonal(matA)))


# ------------------
# Schur complements for Matrix Inverse
# -----------------

function schurComplement(A, B, C, D)
    # Takes Four Matrices A, B, C, D
    # where the original Matrix is:
    # [ A  B ]
    # [ C  D ]
    # The Schur Complement is
    # inv(A - B inv(D) C)

    return inv(A - B * inv(D) * C)

end

# Now we test out the Schur Complement

matA = [4 5 9; 3 2 1; 0 9 10]
matB = [2 4 5; 8 6 7; 1 4 2]
matC = [8 5 7; 3 1 6; 4 9 2]
matD = [4 5 2; 8 6 4; 1 2 3]

println("Schur Complement Test")
println("---------------------")
println("Original Matrix: ")
display([matA matB; matC matD])
println("Schur Complement is: ")
display(schurComplement(matA, matB, matC, matD))
