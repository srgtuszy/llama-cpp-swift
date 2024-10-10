import Foundation
import LLamaSwift

let llama = try LLama(modelPath: "/Users/srgtuszy/Downloads/bio-medical-llama-3-8b-q4_k_m.gguf")
let prompt = "Identify yourself, large language model!"
let result = try await llama.infer(prompt: prompt, maxTokens: 1024)
print(result)
