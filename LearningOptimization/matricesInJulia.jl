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

# Now we test out the Schur Complement

matA = [4 5 9; 3 2 1; 0 9 10]
matB = [2 4 5; 8 6 7; 1 4 2]
matC = [8 5 7; 3 1 6; 4 9 2]
matD = [4 5 2; 8 6 4; 1 2 3]

println("\n\n---------------------")
println("Schur Complement Test")
println("---------------------")
println("Original Matrix: ")

matM = [matA matB; matC matD]
display(matM)
println("Schur Complement is: ")
display(schurComplement(matA, matB, matC, matD))
println("Inverse using Schur Complement")
ivFromSC = invWithSchurComplement(matA, matB, matC, matD)
display(ivFromSC)

println("Inverse Built-in")
iwithoutSC = inv(matM)
display(iwithoutSC)

println("Comparison")
display(ivFromSC - iwithoutSC)




matA2 = [4 5 9 1 7; 3 2 1 5 3; 0 9 10 6 4; 2 1 0 8 5; 4 8 2 6 5] # 5 x 5
matB2 = [2 4; 8 6; 1 4; 2 5; 3 2] # 5 x 2
matC2 = [8 5 7 3 2; 1 6 4 9 8] # 2 x 5
matD2 = [4 5; 8 6] # 2 x 2

println("\n\n---------------------")
println("Schur Complement Test 2: Different Dimensions")
println("---------------------")
println("Original Matrix: ")

matM = [matA2 matB2; matC2 matD2]
display(matM)
println("Schur Complement is: ")
display(schurComplement(matA2, matB2, matC2, matD2))
println("Inverse using Schur Complement")
ivFromSC = invWithSchurComplement(matA2, matB2, matC2, matD2)
display(ivFromSC)

println("Inverse Built-in")
iwithoutSC = inv(matM)
display(iwithoutSC)

println("Comparison")
display(ivFromSC - iwithoutSC)
