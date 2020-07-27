

# This simply holds the feasibility checks
function isFeasiblePolyHedron(A, b, x)
    #=
    Checks if all polyhedron checks are satisfied
    Namely: Ax ≤ b
    =#

    state = A * x - b

    if maximum(state) > 0
        return false
    end

    return true
end


runTests = false

if runTests
    println("\n")
    print("Choosing a point in the feasible region (true): ")
    println(isFeasiblePolyHedron([4 5; -4 5; 0 -1], [20; 30; 1], [-1; 0.5]))

    print("Choosing a point outside the feasible region (false): ")
    println(isFeasiblePolyHedron([4 5; -4 5; 0 -1], [20; 30; 1], [-10; 0.5]))

    print("Choosing another point outside the feasible region (false): ")
    println(isFeasiblePolyHedron([4 5; -4 5; 0 -1], [20; 30; 1], [-1; 10]))

    print("Choosing another point outside the feasible region (false): ")
    println(isFeasiblePolyHedron([4 5; -4 5; 0 -1], [20; 30; 1], [-1; -10]))
end
