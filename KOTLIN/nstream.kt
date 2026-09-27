import kotlin.math.abs
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    println("Parallel Research Kernels.")
    println("Kotlin Stream triad: A = B + scalar * C.")

    /*******************************************************************
    **read and test input parameters
    *******************************************************************/

    if (args.size != 2) {
        println("Usage: kotlin nstream <# iterations> <vector length>")
        exitProcess(1)
    }

    val iterations = args[0].toInt()
    if (iterations < 1) {
        println("ERROR: iterations must be >= 1")
        exitProcess(1)
    }

    val length = args[1].toInt()
    if (length < 1) {
        println("ERROR: vector length must be positive")
        exitProcess(1)
    }

    println("Vector length        = $length")
    println("Number of iterations = $iterations")

    val a = DoubleArray(length)
    val b = DoubleArray(length) { 2.0 }
    val c = DoubleArray(length) { 2.0 }

    /* --- MAIN LOOP --- repeat Triad iterations times --- */

    val scalar = 3.0

    var startTime = 0L

    for (iter in 0..iterations) {

        /* start timer after a warmup iteration */
        if (iter == 1)
            startTime = System.currentTimeMillis()

        for (j in 0 until length)
            a[j] += b[j] + scalar * c[j]
    }

    /*********************************************************************
    ** Analyze and output results.
    *********************************************************************/

    val streamTime = System.currentTimeMillis() - startTime

    var ar = 0.0
    val br = 2.0
    val cr = 2.0

    for (k in 0..iterations)
        ar += br + scalar * cr

    ar *= length

    var asum = 0.0
    for (i in 0 until length)
        asum += abs(a[i])

    val epsilon = 1.0e-8
    if (abs(ar - asum) / asum > epsilon) {
        println("Failed Validation on output array")
        println("        Expected checksum: $ar")
        println("        Observed checksum: $asum")
        println("ERROR: solution did not validate")
    } else {
        println("Solution validates")
        val avgtime = streamTime / iterations.toDouble() / 1000
        val nbytes = 4.0 * length * 8
        println(String.format("Rate (MB/s): %f Avg time (s): %f", 1.0e-6 * nbytes / avgtime, avgtime))
    }
}
