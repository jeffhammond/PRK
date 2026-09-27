import kotlin.math.abs
import kotlin.math.min
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    println("Parallel Research Kernels.")
    println("Kotlin pipeline execution on 2D grid.")

    /*******************************************************************
    **read and test input parameters
    *******************************************************************/
    if (args.size < 2 || args.size > 5) {
        println("Usage: kotlin stencil <# iterations> <array dimension> [<star/stencil> <radius> <tile size>]")
        exitProcess(1)
    }

    val iterations = args[0].toInt()
    if (iterations < 1) {
        println("ERROR: iterations must be >= 1")
        exitProcess(1)
    }

    val n = args[1].toInt()
    if (n < 1) {
        println("ERROR: grid dimension must be positive: $n")
        exitProcess(1)
    }

    val pattern = if (args.size > 2) args[2] else "star"

    val radius: Int
    if (args.size > 3) {
        radius = args[3].toInt()
        if (radius < 1) {
            println("ERROR: Stencil radius should be positive.")
            exitProcess(1)
        }
        if (2 * radius + 1 > n) {
            println("ERROR: Stencil radius exceeds grid size")
            exitProcess(1)
        }
    } else {
        radius = 2
    }

    var tileSize = 0
    if (args.size > 4) {
        tileSize = args[4].toInt()
        if (tileSize <= 0 || tileSize > n)
            tileSize = n
    }

    println("Number of iterations = $iterations")
    println("Grid size            = $n")

    if (pattern == "star")
        println("Type of stencil      = star")
    else
        println("Type of stencil      = stencil")

    if (tileSize != n)
        println("Tile size            = $tileSize")
    else
        println("Untiled")

    println("radius of stencil    = $radius")
    println("Data type            = double precision")
    println("Compact representation of stencil loop body")

    // initialize the input, output, and weight arrays
    val weight = Array(2 * radius + 1) { DoubleArray(2 * radius + 1) }
    val input = Array(n) { i -> DoubleArray(n) { j -> (i + j).toDouble() } }
    val out = Array(n) { DoubleArray(n) }

    val stencilSize: Int
    if (pattern == "star") {
        stencilSize = 4 * radius + 1
        for (i in 1..radius) {
            weight[radius][radius + i] = +1.0 / (2 * i * radius)
            weight[radius + i][radius] = +1.0 / (2 * i * radius)
            weight[radius][radius - i] = -1.0 / (2 * i * radius)
            weight[radius - i][radius] = -1.0 / (2 * i * radius)
        }
    } else {
        stencilSize = (2 * radius + 1) * (2 * radius + 1)
        for (j in 1..radius) {
            for (i in -j + 1 until j) {
                weight[radius + i][radius + j] = +1.0 / (4 * j * (2 * j - 1) * radius)
                weight[radius + i][radius - j] = -1.0 / (4 * j * (2 * j - 1) * radius)
                weight[radius + j][radius + i] = +1.0 / (4 * j * (2 * j - 1) * radius)
                weight[radius - j][radius + i] = -1.0 / (4 * j * (2 * j - 1) * radius)
            }
            weight[radius + j][radius + j] = +1.0 / (4 * j * radius)
            weight[radius - j][radius - j] = -1.0 / (4 * j * radius)
        }
    }

    var startTime = 0L
    var normal = 0.0
    val activePoints = ((n - 2 * radius) * (n - 2 * radius)).toDouble()
    val epsilon = 1.0e-8
    val referenceNorm = 2 * (iterations + 1)

    for (iter in 0..iterations) {

        // start timer after a warmup iteration
        if (iter == 1)
            startTime = System.currentTimeMillis()

        // Apply the stencil operator

        if (tileSize == 0) {
            for (j in radius until n - radius) {
                for (i in radius until n - radius) {
                    if (pattern == "star") {
                        for (jj in -radius..radius)
                            out[j][i] += weight[0 + radius][jj + radius] * input[j + jj][i]
                        for (ii in -radius until 0)
                            out[j][i] += weight[ii + radius][0 + radius] * input[j][i + ii]
                        for (ii in 1..radius)
                            out[j][i] += weight[ii + radius][0 + radius] * input[j][i + ii]
                    } else {
                        for (jj in -radius..radius) {
                            for (ii in -radius..radius)
                                out[j][i] += weight[ii + radius][jj + radius] * input[j + jj][i + ii]
                        }
                    }
                }
            }
        } else {
            var jt = radius
            while (jt < n - radius) {
                var it = radius
                while (it < n - radius) {
                    for (j in jt until min(n - radius, jt + tileSize)) {
                        for (i in it until min(n - radius, it + tileSize)) {
                            if (pattern == "star") {
                                for (jj in -radius..radius)
                                    out[j][i] += weight[0 + radius][jj + radius] * input[j + jj][i]
                                for (ii in -radius until 0)
                                    out[j][i] += weight[ii + radius][0 + radius] * input[j][i + ii]
                                for (ii in 1..radius)
                                    out[j][i] += weight[ii + radius][0 + radius] * input[j][i + ii]
                            } else {
                                for (jj in -radius..radius) {
                                    for (ii in -radius..radius)
                                        out[j][i] += weight[ii + radius][jj + radius] * input[j + jj][i + ii]
                                }
                            }
                        }
                    }
                    it += tileSize
                }
                jt += tileSize
            }
        }
        // add constant to solution to force refresh of neighbor data, if any
        for (j in 0 until n) {
            for (i in 0 until n)
                input[j][i] += 1.0
        }
    }

    /*********************************************************************
    ** Analyze and output results.
    *********************************************************************/

    val stencilTime = System.currentTimeMillis() - startTime

    for (j in radius until n - radius) {
        for (i in radius until n - radius) {
            normal += abs(out[j][i])
        }
    }

    normal /= activePoints

    if (abs(normal - referenceNorm) < epsilon) {
        println("Solution validates")
        val flops = (2 * stencilSize + 1) * activePoints
        val avgtime = stencilTime / iterations.toDouble() / 1000
        println(String.format("Rate (MFlops/s): %f Avg time (s):%f", 1.0e-6 * flops / avgtime, avgtime))
    } else {
        println(String.format("ERROR: L1 norm = %.10f Reference L1 norm = %d", normal, referenceNorm))
    }
}
