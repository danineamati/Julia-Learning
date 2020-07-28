# Run the tests on the projections

include("..\\src\\constraints\\projections.jl")



runTests = true

if runTests
    println("\n")
    print("Testing projection onto a line: ")
    vTest = [1; 3]
    ptTest = [0; -2]
    projptTest = projLine(ptTest, vTest)
    print("$projptTest ?= [-0.6; -1.8] : Passed? ")
    println(" $(projptTest == [-0.6; -1.8])")
    optTest = round.(orthoProjLine(ptTest, vTest), digits=5)
    print("Orthogonal projection onto a line: ")
    print("$optTest ?= [0.6; -0.2] : Passed? ")
    println(" $(optTest == [0.6; -0.2])")

    println()
    print("Now we try a projection onto the positive orthant: ")
    pt2Test = [5; -3; -100; 0]
    projpt2Test = projPosOrth(pt2Test)
    println("$pt2Test -> $projpt2Test : Passed? $(projpt2Test .≥ 0)")

    println()
    print("Next up is an affine constraint: ")
    aTest = [5; 4; 3]
    bTest = 4
    pt3Test = [10; 10; 17]
    projpt3Test = projAffineEq(aTest, bTest, pt3Test)
    xp = pt3Test - projpt3Test
    println("$pt3Test -> $projpt3Test")
    print("On Hyperplane : $(aTest'projpt3Test == bTest), ")
    println("Min Dist : $(xp'aTest == norm(xp) * norm(aTest))")

    println()
    println("Lastly is the second order cone")
    println("The first cone is the simplest unit cone at the origin")
    sTest1 = 6
    pt4Test = [3; 4; 5]
    projpt4Test = projSecondOrderCone(pt4Test, sTest1)
    print("$pt4Test, $sTest1 -> $projpt4Test, ")
    println("Base norm = $(norm(pt4Test))")
    sTest2 = 10
    projpt4Test = projSecondOrderCone(pt4Test, sTest2)
    print("$pt4Test, $sTest1 -> $projpt4Test, ")
    println("Base norm = $(norm(pt4Test))")
    sTest3 = -10
    projpt4Test = projSecondOrderCone(pt4Test, sTest2)
    print("$pt4Test, $sTest1 -> $projpt4Test, ")
    println("Base norm = $(norm(pt4Test))")

    print("The second cone now is offeset with v -> Ax - b -> ")
    AMat = [4 5; -4 5; 0 -1]
    bVec = [20; 30; 1]
    xTest = [3; 4]
    sTest2 = 10
    projpt5Test = projSecondOrderCone(AMat * xTest - bVec, sTest2)
    println(" $(AMat * xTest - bVec) for x = $xTest")
    println("Proj -> $projpt5Test")

end
