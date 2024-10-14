import Foundation
import LLamaSwift

let model = try Model(modelPath: "/Users/srgtuszy/Downloads/bio-medical-llama-3-8b-q4_k_m.gguf")
let llama = LLama(model: model)
let prompt = "what is the meaning of life?"

for try await token in await llama.infer(prompt: prompt, maxTokens: 1024) {
    print(token, terminator: "")
}
