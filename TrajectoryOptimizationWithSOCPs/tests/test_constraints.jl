# Run the tests on the constraints

include("..\\src\\constraints\\constraints.jl")

runTestsAffine = false
runTestsOldCone = false # NOT CURRENTLY WORKING AND NOT USED
runTestsNewCone = true

if runTestsAffine
    println()
    println("Affine Equality Constraints:")
    dT = AL_AffineEquality([5], [10])
    println("** Simplest")
    println("On line  (True) : $(satisfied(dT, 2))")
    println("Off line (False): $(satisfied(dT, 4)) and $(satisfied(dT, 0))")
    dT2 = AL_AffineEquality([5 0; 0 6], [10; 12])
    println("** 2D")
    println("On Intesection (True)        : $(satisfied(dT2, [2; 2]))")
    println("Check which not (True, False): $(whichSatisfied(dT2, [2; 4]))")
    println("** Testing the projections and violations")
    testpt = [10; 12]
    println("Point $testpt not at intersetion : $(!satisfied(dT2, testpt))")
    println("Projection onto constraints : $(getProjVecs(dT2, testpt))")
    gN = getNormToProjVals(dT2, testpt)
    println("Violation is : $gN")
    println("Total Violation is : $(gN'gN)")
    println("** Testing Calculus")
    println("Grad = $(getGradC(dT2)) and Hessian = $(getHessC(dT2))")

    println("\nAffine Inequality Constraints")
    diT = AL_AffineInequality([5], [10])
    println("** Simplest")
    println("On line    (True) : $(satisfied(diT, 2))")
    println("Above line (False): $(satisfied(diT, 4))")
    println("Under line (True) : $(satisfied(diT, 0))")
    diT2 = AL_AffineInequality([5 0; 0 6], [10; 12])
    println("** 2D")
    println("On Intesection (True)        : $(satisfied(diT2, [2; 2]))")
    println("Check which not (True, False): $(whichSatisfied(diT2, [2; 4]))")
    println("Check which not (True, True) : $(whichSatisfied(diT2, [2; 0]))")
    println("** Testing the projections and violations")
    testpt = [10; 12]
    println("Point $testpt not at in region : $(!satisfied(diT2, testpt))")
    println("Projection onto feasible region : $(getProjVecs(diT2, testpt))")
    gN = getNormToProjVals(diT2, testpt)
    println("Violation is : $gN")
    println("Total Violation is : $(gN'gN)")
    testpt2 = [-10; -12]
    println("Point $testpt2 in feasible region : $(satisfied(diT2, testpt2))")
    println("Projection onto feasible region : $(getProjVecs(diT2, testpt2))")
    gN2 = getNormToProjVals(diT2, testpt2)
    println("Violation is : $gN2")
    println("Total Violation is : $(gN2'gN2)")
    println("** Testing Calculus")
    println("Hessian = $(getHessC(diT2))")
    println("Grad at $testpt = $(getGradC(diT2, testpt))")
    println("Grad at $testpt2 = $(getGradC(diT2, testpt2))")
end

if runTestsOldCone
    println("\nSecond-Order Constraints")
    dconeT = AL_pCone([5][:,:], vcat([-4]), vcat([3]), -8, 2)
    xRan = -5:5
    println("\n** Simplest Full")
    println("xVals = $(collect(xRan))")
    println("Raw Vals = $([getRaw(dconeT, vcat([x])) for x in xRan])")
    println("Satisfied = $([satisfied(dconeT, vcat([x])) for x in xRan])")
    println("Projection = $([getProjVecs(dconeT, vcat([x])) for x in xRan])")
    println("Violation = $([getNormToProjVals(dconeT, vcat([x])) for x in xRan])")

    dconeT2D = AL_pCone([5 3; 3 4], [-4; -2], [3; 4], -8, 2)
    xRan = -4:5
    println("\n** 2D Full")
    println("xVals = $(collect(xRan))")
    println("In the form [x; x]")
    println("Raw Vals = $([getRaw(dconeT2D, [x; x]) for x in xRan])")
    println("Satisfied = $([satisfied(dconeT2D, [x; x]) for x in xRan])")
    projVecList = [getProjVecs(dconeT2D, [x; x], true) for x in xRan]
    println("Projection = ")
    display(projVecList)
    println("Violation = $([getNormToProjVals(dconeT2D, [x; x]) for x in xRan])")
    println("Gradients = ")
    gradVecs = [getGradC(dconeT2D, [x; x]) for x in xRan]
    display(gradVecs)
    println("Hessians = ")
    hessVecs = [getHessC(dconeT2D, [x; x]) for x in xRan]
    display(hessVecs)
end

if runTestsNewCone
    dslackCone1 = AL_coneSlack([-1 1], [0], [1; 1], 0)
    #AL_coneSlack([2 1/3; 1/3 3/4; 1 1], [3; 4; 5], [2; 1], -8)

    println("\n** Cone With Slack Variables")
    display(dslackCone1)

    xRan = -1:1
    s = [6] #[2; 5; -4]
    t = 5
    println("Probing: (with s = $s and t = $t): ")
    println([[x; x] for x in xRan])
    println("Raw Values: ")
    println([getRaw(dslackCone1, [x; x], s, t) for x in xRan])
    println("Satisfied: ")
    println([whichSatisfied(dslackCone1, [x; x], s, t) for x in xRan])
    println("Projections: ")
    println([getProjVecs(dslackCone1, [x; x], s, t) for x in xRan])
    println("Violations: ")
    println([getNormToProjVals(dslackCone1, [x; x], s, t) for x in xRan])

    println("\n\nGradient")
    xtest = [-5; 6]
    stest = [10] #dslackCone1.A * xtest
    ttest = 5 #dslackCone1.c' * xtest
    println("Probing: x = $xtest, s = $stest, t = $ttest")
    print("Which satisfied: ")
    println(whichSatisfied(dslackCone1, xtest, stest, ttest))
    jc = getGradC(dslackCone1, xtest, stest, ttest, true)
    display(reshape(jc, size(jc)))
    print("Violation: ")
    cCurr = getNormToProjVals(dslackCone1, xtest, stest, ttest)
    println(cCurr)
    println("Overall: -j'c")
    println(-jc'*cCurr)

    println("\n\nHessian")
    hc = getHessC(dslackCone1, xtest, stest, ttest)
    display(hc)

    if true
        dslackCone2 = AL_coneSlack([2 1/3; 1/3 3/4; 1 1], [3; 4; 5], [2; 1], -8)
        xtest = [-5; 6]
        stest = dslackCone2.A * xtest
        ttest = dslackCone2.c' * xtest
        println("\n\nLarge Hessian")
        hc = getHessC(dslackCone2, xtest, stest, ttest)
        display(hc)
    end

end


println("Tests Complete")
