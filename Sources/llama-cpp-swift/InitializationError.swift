struct InitializationError: Error, Sendable {
    let message: String
    let code: Code
    
    enum Code: Int, Sendable {
        case failedToLoadModel = 1
        case failedToInitializeContext
    }
}
