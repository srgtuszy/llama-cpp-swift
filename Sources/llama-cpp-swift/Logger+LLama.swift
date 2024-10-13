import Logging

extension Logger {
    static var llama: Logger {
        Logger(label: "llama-cpp-swift")
    }
}
