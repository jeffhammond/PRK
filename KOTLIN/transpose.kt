import kotlin.math.abs
import kotlin.math.min
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    println("Parallel Research Kernels.")
    println("Kotlin Matrix transpose: B = A^T.")

    /*******************************************************************
    **read and test input parameters
    *******************************************************************/

    if (args.size != 3 && args.size != 2) {
        println("Usage: kotlin transpose <# iterations> <matrix order> [tile size]")
        exitProcess(1)
    }

    val iterations = args[0].toInt()
    if (iterations < 1) {
        println("ERROR: iterations must be >= 1")
        exitProcess(1)
    }

    val order = args[1].toInt()
    if (order < 1) {
        println("ERROR: matrix order must be >= 1")
        exitProcess(1)
    }

    var tileSize = if (args.size == 3) args[2].toInt() else 0
    /* a non-positive tile size means no tiling of the local transpose */
    if (tileSize <= 0)
        tileSize = order

    println("Matrix order          = $order")
    if (tileSize < order)
        println("Tile size             = $tileSize")
    else
        println("Untiled")
    println("Number of iterations  = $iterations")

    val a = Array(order) { i -> DoubleArray(order) { j -> (i * order + j).toDouble() } }
    val b = Array(order) { DoubleArray(order) }

    var startTime = 0L
    val epsilon = 1.0e-8
    val bytes = 2.0 * 8 * order * order

    for (iter in 0..iterations) {

        /* start timer after a warmup iteration                                        */
        if (iter == 1)
            startTime = System.currentTimeMillis()

        /* Transpose the  matrix; only use tiling if the tile size is smaller
           than the matrix */
        if (tileSize < order) {
            var i = 0
            while (i < order) {
                var j = 0
                while (j < order) {
                    for (it in i until min(order, i + tileSize)) {
                        for (jt in j until min(order, j + tileSize)) {
                            b[it][jt] += a[jt][it]
                            a[jt][it] += 1.0
                        }
                    }
                    j += tileSize
                }
                i += tileSize
            }
        } else {
            for (i in 0 until order)
                for (j in 0 until order) {
                    b[i][j] += a[j][i]
                    a[j][i] += 1.0
                }
        }
    }

    /*********************************************************************
    ** Analyze and output results.
    *********************************************************************/

    val transposeTime = System.currentTimeMillis() - startTime

    var abserr = 0.0
    val addit = ((iterations + 1).toDouble() * iterations.toDouble()) / 2.0
    for (j in 0 until order) {
        for (i in 0 until order) {
            abserr += abs(b[j][i] - ((order * i + j).toDouble() * (iterations + 1) + addit))
        }
    }

    if (abserr < epsilon) {
        println("Solution validates")
        val avgtime = transposeTime / iterations.toDouble() / 1000
        println(String.format("Rate (MB/s): %f Avg time (s): %f", 1.0e-6 * bytes / avgtime, avgtime))
    } else {
        println(String.format("ERROR: Aggregate squared error %f exceeds threshold %f", abserr, epsilon))
    }
}
