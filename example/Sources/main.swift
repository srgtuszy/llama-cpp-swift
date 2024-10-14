import Foundation
import LLamaSwift

guard let modelPath = ProcessInfo.processInfo.environment["MODEL_PATH"] else {
    print("Error: MODEL_PATH environment variable not set.")
    exit(1)
}

let model = try Model(modelPath: modelPath)
let llama = LLama(model: model)
let prompt = "what is the meaning of life?"

for try await token in await llama.infer(prompt: prompt, maxTokens: 1024) {
    print(token, terminator: "")
}
