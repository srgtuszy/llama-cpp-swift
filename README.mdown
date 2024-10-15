# llama-cpp-swift
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fsrgtuszy%2Fllama-cpp-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/srgtuszy/llama-cpp-swift) [![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fsrgtuszy%2Fllama-cpp-swift%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/srgtuszy/llama-cpp-swift)

Swift bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) thanks to which you'll be able to run compatible LLM models directly on your device.

## Features

- Lightweight and easy to use
- Works on macOS and Linux
- Supports streaming via structured concurrency
- Swift 6 ready!

## TODO

- [ ] Unit tests
- [ ] Model downloads from URL and HuggingFace

## How to install

Use swift package manager:

```
.package(url: "https://github.com/srgtuszy/llama-cpp-swift", branch: "main")
```

## How to use

Here's a quick example on how to use it. For more, please refer to an example app in `example/` folder.

```swift
// Initialize model
let model = try Model(modelPath: "<model path>")
let llama = try LLama(model: model)

// Results are delivered through an `AsyncStream`
let prompt = "what is the meaning of life?"
for try await token in await llama.infer(prompt: prompt, maxTokens: 1024) {
    print(token, terminator: "")
}
```
