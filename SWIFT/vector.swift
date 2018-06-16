import Foundation
import Accelerate
import MetalPerformanceShaders

let N = 1000

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let commandBuffer = commandQueue.makeCommandBuffer()!

let buffer = device.makeBuffer(length: N * MemoryLayout<Float32>.size)
