import Foundation
import LLamaSwift

let llama = try LLama(modelPath: "/Users/srgtuszy/Downloads/bio-medical-llama-3-8b-q4_k_m.gguf")
let prompt = "what is the meaning of life?"

for try await token in llama.infer(prompt: prompt, maxTokens: 1024) {
    print(token, terminator: "")
}
