import Foundation
import Metal
import CryptoKit

struct Params {
    var iterations: UInt32
    var tile: UInt32
    var rounds: UInt32
    var stride: UInt32
    var a: Float
    var b: Float
    var eta: Float
    var d: Float
}

struct Config: Hashable {
    var kernel: String
    var tile: Int
    var width: Int
    var groups: Int
}

struct Generator: RandomNumberGenerator {
    var value: UInt64 = 20261107
    mutating func next() -> UInt64 {
        value = value &* 6364136223846793005 &+ 1442695040888963407
        return value
    }
}

func emit(_ value: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
    fflush(stdout)
}

enum ProbeError: Error { case failed(String) }
let args = CommandLine.arguments
guard args.count >= 2 else { fatalError("benchmark source.metal [sweep|quick|profile] [repetitions=5] [iterations=4096]") }
let mode = args.count > 2 ? args[2] : "sweep"
let repetitions = args.count > 3 ? Int(args[3])! : 5
let iterations = args.count > 4 ? Int(args[4])! : 4096
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let options = MTLCompileOptions()
options.fastMathEnabled = false
let library = try device.makeLibrary(source: String(contentsOfFile: args[1], encoding: .utf8), options: options)
var pipelines: [String: MTLComputePipelineState] = [:]
for name in ["fma1", "fma4", "rr_uniform", "rr_matrix"] {
    let p = try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
    guard p.maxTotalThreadsPerThreadgroup >= 1024 else { throw ProbeError.failed("Pipeline does not support W1024: \(name)") }
    pipelines[name] = p
}
try emit([
    "kind": "metadata", "device": device.name,
    "max_threadgroup_memory": device.maxThreadgroupMemoryLength,
    "counter_sets": (device.counterSets ?? []).map { ["name": $0.name, "counters": $0.counters.map { $0.name }] },
    "pipelines": pipelines.mapValues { ["max_threads": $0.maxTotalThreadsPerThreadgroup,
                                         "simd_width": $0.threadExecutionWidth,
                                         "static_shared_bytes": $0.staticThreadgroupMemoryLength] },
    "mode": mode, "repetitions": repetitions, "iterations": iterations,
    "timing": "MTLCommandBuffer.gpuEndTime - gpuStartTime; serial compute dispatches",
    "fast_math": false
])

let maxGroups = 256
let maxThreads = maxGroups * 1024
var inputValues = [SIMD4<Float>]()
for i in 0..<maxThreads {
    let f = Float(i % 1024)
    inputValues.append(SIMD4<Float>(0.1 + f * 0.0002, 0.2 + f * 0.00013,
                                   0.7 - f * 0.0001, 0.8 - f * 0.00017))
}
let initial = inputValues.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
let output = device.makeBuffer(length: maxThreads * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!
let matrixLength = 8192 * 1024
let distanceBuffer = device.makeBuffer(length: matrixLength * MemoryLayout<Float>.stride, options: .storageModeShared)!
distanceBuffer.contents().bindMemory(to: Float.self, capacity: matrixLength).initialize(repeating: 1, count: matrixLength)

let groupCounts = mode == "boundary" ?
    [1, 4, 8, 16, 20, 21, 24, 32, 40, 41, 64, 80, 81, 128, 160, 161] :
    [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 64, 80, 128, 256]
let widths = [32, 64, 128, 256, 512, 1024]
var configs: [Config] = []
for width in widths {
    for groups in groupCounts {
        for name in ["fma1", "fma4"] {
            configs.append(Config(kernel: name, tile: width, width: width, groups: groups))
        }
        for tileWidth in Set([width, 1024]).sorted() {
            for name in ["rr_uniform", "rr_matrix"] {
                configs.append(Config(kernel: name, tile: width, width: tileWidth, groups: groups))
            }
        }
    }
}
if mode == "quick" {
    configs = configs.filter { [128, 1024].contains($0.tile) && [1, 4, 20, 64].contains($0.groups) }
}
if mode == "profile" {
    configs = [Config(kernel:"fma4",tile:128,width:128,groups:20),
               Config(kernel:"fma4",tile:1024,width:1024,groups:20),
               Config(kernel:"rr_uniform",tile:128,width:1024,groups:20),
               Config(kernel:"rr_uniform",tile:1024,width:1024,groups:20)]
}
if mode == "boundary" {
    configs = configs.filter {
        $0.kernel == "fma4" || ($0.kernel == "rr_uniform" && $0.tile != 512)
    }
}
if mode == "profile2" {
    configs = []
    for groups in [1, 20, 40, 60, 80, 120] {
        for width in [128, 256, 1024] {
            configs.append(Config(kernel:"fma4",tile:width,width:width,groups:groups))
        }
        for tile in [128, 1024] {
            configs.append(Config(kernel:"rr_uniform",tile:tile,width:1024,groups:groups))
        }
    }
}
if mode == "resident" {
    configs = []
    // Enough queued work to expose the observed residency ceiling for each W.
    // 131072 output elements fit the existing 262144-element allocation.
    for width in widths {
        for kernel in ["fma4", "rr_uniform"] {
            configs.append(Config(kernel:kernel,tile:width,width:width,groups:131072/width))
        }
    }
}
if mode == "fixed" {
    // Hold total threads and per-thread arithmetic fixed; vary only WG geometry.
    configs = []
    for count in [1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144] {
        for width in widths {
            for kernel in ["fma1", "fma4"] {
                configs.append(Config(kernel:kernel,tile:width,width:width,groups:count/width))
            }
        }
    }
}
if mode == "rr-fixed" {
    configs = []
    // At each (T,N), keep WG count, shared bytes, barriers and updates fixed.
    // Only change threads per WG, including W<T and W>T controls.
    for count in [8192, 32768, 131072] {
        for tile in [64, 128, 256, 512, 1024] {
            for width in widths {
                for kernel in ["rr_uniform", "rr_matrix"] {
                    configs.append(Config(kernel:kernel,tile:tile,width:width,groups:count/tile))
                }
            }
        }
    }
}
var rng = Generator()
var outputHashes: [String: String] = [:]
let dispatches = 4
let rounds = 4

for repetition in 0...repetitions {
    configs.shuffle(using: &rng)
    for config in configs {
        try autoreleasepool {
            let pipeline = pipelines[config.kernel]!
            let isRR = config.kernel.hasPrefix("rr_")
            let count = config.groups * (isRR ? config.tile : config.width)
            var params = Params(iterations: UInt32(iterations), tile: UInt32(config.tile),
                                rounds: UInt32(rounds), stride: 8192,
                                a: 0.99991, b: 0.000031, eta: 0.01, d: 1)
            let cb = queue.makeCommandBuffer()!
            cb.label = "\(config.kernel) T\(config.tile) W\(config.width) G\(config.groups) r\(repetition)"
            let encoder = cb.makeComputeCommandEncoder(dispatchType: .serial)!
            encoder.label = cb.label
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(initial, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
            if isRR {
                encoder.setBuffer(distanceBuffer, offset: 0, index: 3)
                encoder.setThreadgroupMemoryLength(config.tile * 16, index: 0)
            }
            for _ in 0..<dispatches {
                encoder.dispatchThreadgroups(MTLSize(width: config.groups, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: config.width, height: 1, depth: 1))
            }
            encoder.endEncoding()
            let wallStart = DispatchTime.now().uptimeNanoseconds
            cb.commit()
            cb.waitUntilCompleted()
            let wallMS = Double(DispatchTime.now().uptimeNanoseconds - wallStart) / 1e6
            if cb.status != .completed { throw cb.error ?? ProbeError.failed("GPU command failed") }
            let gpuMS = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            guard gpuMS > 0 else { throw ProbeError.failed("Invalid GPU timing") }
            let raw = UnsafeRawBufferPointer(start: output.contents(), count: count * MemoryLayout<SIMD4<Float>>.stride)
            let digest = SHA256.hash(data: Data(raw)).map { String(format: "%02x", $0) }.joined()
            let data = output.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
            guard data[0].x.isFinite && data[count-1].x.isFinite else { throw ProbeError.failed("Nonfinite result") }
            // Same total scalar-thread work, independently of grouping.
            let key: String
            let checkDigest: String
            if isRR {
                key = "rr-T\(config.tile)"
                let first = Data(bytes: output.contents(), count: config.tile * MemoryLayout<SIMD4<Float>>.stride)
                checkDigest = SHA256.hash(data: first).map { String(format: "%02x", $0) }.joined()
                for group in 1..<config.groups {
                    let other = Data(bytes: output.contents().advanced(by: group * first.count), count: first.count)
                    guard first == other else { throw ProbeError.failed("RR groups differ") }
                }
            } else {
                key = "\(config.kernel)-N\(count)"
                checkDigest = digest
                for index in [0, count / 2, count - 1] {
                    var expected = inputValues[index]
                    for _ in 0..<iterations {
                        for lane in 0..<(config.kernel == "fma4" ? 4 : 1) {
                            expected[lane] = params.b.addingProduct(expected[lane], params.a)
                        }
                    }
                    if config.kernel == "fma1" { expected = SIMD4<Float>(expected.x, 0, 0, 0) }
                    guard data[index] == expected else { throw ProbeError.failed("CPU FMA mismatch at \(index)") }
                }
            }
            if let previous = outputHashes[key] {
                guard previous == checkDigest else { throw ProbeError.failed("Regrouping output mismatch: \(key)") }
            } else { outputHashes[key] = checkDigest }
            let chains = config.kernel == "fma4" ? 4 : 1
            let work = isRR ? config.groups * config.tile * config.tile * rounds * dispatches :
                count * iterations * chains * dispatches
            try emit([
                "kind": "measurement", "kernel": config.kernel,
                "tile": config.tile, "width": config.width, "groups": config.groups,
                "active_threads": isRR ? config.tile : config.width,
                "allocated_threads_per_dispatch": config.width * config.groups,
                "output_count": count, "iterations": iterations, "rr_rounds": rounds,
                "dispatches": dispatches, "work_items": work,
                "work_unit": isRR ? "pair_updates" : "scalar_fma",
                "gpu_ms": gpuMS, "wall_ms": wallMS,
                "repetition": repetition, "warmup": repetition == 0,
                "output_sha256": digest, "equivalence_sha256": checkDigest,
                "shared_bytes_per_group": isRR ? config.tile * 16 : 0
            ])
        }
    }
}
