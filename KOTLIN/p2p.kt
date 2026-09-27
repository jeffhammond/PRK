import kotlin.math.abs
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    println("Parallel Research Kernels.")
    println("Kotlin pipeline execution on 2D grid.")

    /*******************************************************************
    **read and test input parameters
    *******************************************************************/

    if (args.size != 3) {
        println("argument count = ${args.size}")
        println("Usage: kotlin p2p <# iterations> <first array dimension> <second array dimension>")
        exitProcess(1)
    }

    val iterations = args[0].toInt()
    if (iterations < 1) {
        println("ERROR: iterations must be >= 1")
        exitProcess(1)
    }

    val m = args[1].toInt()
    if (m < 1) {
        println("ERROR: array dimension must be >= 1")
        exitProcess(1)
    }

    val n = args[2].toInt()
    if (n < 1) {
        println("ERROR: array dimension must be >= 1")
        exitProcess(1)
    }

    println(String.format("Grid sizes               = %d * %d", m, n))
    println(String.format("Number of iterations     = %d", iterations))

    val grid = Array(m) { DoubleArray(n) }
    for (j in 0 until n)
        grid[0][j] = j.toDouble()
    for (i in 0 until m)
        grid[i][0] = i.toDouble()

    var startTime = 0L
    for (k in 0..iterations) {
        // start timer after a warmup iteration
        if (k < 1)
            startTime = System.currentTimeMillis()

        for (i in 1 until m) {
            for (j in 1 until n) {
                grid[i][j] = grid[i - 1][j] + grid[i][j - 1] - grid[i - 1][j - 1]
            }
        }

        // copy top right corner value to bottom left corner to create dependency
        grid[0][0] = -grid[m - 1][n - 1]
    }

    val pipelineTime = System.currentTimeMillis() - startTime

    /*******************************************************************
    **Final analysis
    *******************************************************************/

    val cornerValue = ((iterations + 1) * (m + n - 2)).toDouble()
    val epsilon = 1.0e-8
    if (abs(grid[m - 1][n - 1] - cornerValue) / cornerValue < epsilon) {
        println("Solution validates")
        val avgtime = pipelineTime / iterations.toDouble() / 1000
        println(String.format("Rate (MFlops/s): %f;  Avg time (s): %f", (1.0e-6) * 2 * (m - 1) * (n - 1) / avgtime, avgtime))
    } else {
        println(String.format("ERROR: checksum %f does not match verification value %f", grid[m - 1][n - 1], cornerValue))
    }
}
