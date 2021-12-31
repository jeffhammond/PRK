import ArgumentParser

import Foundation
import Accelerate
import MetalPerformanceShaders

if CommandLine.arguments.count > 2 {
	print(CommandLine.arguments[1])
	print(CommandLine.arguments[2])
}
let N = 1000

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let commandBuffer = commandQueue.makeCommandBuffer()!

let A = device.makeBuffer(length: N * MemoryLayout<Float64>.size)
let B = device.makeBuffer(length: N * MemoryLayout<Float64>.size)
let C = device.makeBuffer(length: N * MemoryLayout<Float64>.size)
