#
# Copyright (c) 2020, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#*******************************************************************
#
# NAME:    nstream
#
# PURPOSE: To compute memory bandwidth when adding a vector of a given
#          number of double precision values to the scalar multiple of
#          another vector of the same length, and storing the result in
#          a third vector.
#
# USAGE:   The program takes as input the number
#          of iterations to loop over the triad vectors, the length of the
#          vectors, and the offset between vectors
#
#          <progname> <# iterations> <vector length> <offset>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# NOTES:   Bandwidth is determined as the number of words read, plus the
#          number of words written, times the size of the words, divided
#          by the execution time. For a vector length of N, the total
#          number of words read and written is 4*N*sizeof(double).
#
#
# HISTORY: This code is loosely based on the Stream benchmark by John
#          McCalpin, but does not follow all the Stream rules. Hence,
#          reported results should not be associated with Stream in
#          external publications
#
#          Converted to Python by Jeff Hammond, October 2017.
#
# *******************************************************************

# ********************************************************************
# read and test input parameters
# ********************************************************************

import times
#import os

echo "Parallel Research Kernels"
echo "NIM STREAM triad: A = B + scalar * C"

const iterations {.intdefine.}: int = 10
const length     {.intdefine.}: int = 1000

echo "Number of iterations = ", iterations
echo "Vector length        = ", length

# ********************************************************************
# ** Allocate space for the input and execute STREAM triad
# ********************************************************************

type vector = array[1..length, float]

var A : vector
var B : vector
var C : vector

for i in 1..length:
    A[i] = float(0)
    B[i] = float(2)
    C[i] = float(2)

let scalar : float = 3

var t0 : float

for k in 0..iterations:
    if k<1:
        t0 = cpuTime()
    for i in 1..length:
        A[i] += B[i] + scalar * C[i]


let t1 = cpuTime()
let nstream_time = t1 - t0

# ********************************************************************
# ** Analyze and output results.
# ********************************************************************

var ar : float = 0.0
let br : float = 2.0
let cr : float = 2.0

for k in 0..iterations:
    ar += br + scalar * cr

ar = ar * float(length)

var asum : float = 0.0
for i in 1..length:
    asum += abs(A[i])

let epsilon : float = 0.0000001
if abs(ar-asum)/asum > epsilon:
    echo "Failed Validation on output array"
    echo "        Expected checksum: ", ar
    echo "        Observed checksum: ", asum
    echo "ERROR: solution did not validate"
else:
    echo "Solution validates"
    let avgtime : float = float(nstream_time) / float(iterations)
    let nbytes  : float = 4.0 * float(length) * 8.0
    echo "Rate (MB/s): ", 0.000001*nbytes/avgtime, " Avg time (s): ", avgtime


