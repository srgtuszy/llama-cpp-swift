struct InitializationError: Error {
    let message: String
    let code: Code
    
    enum Code: Int {
        case failedToLoadModel = 1
        case failedToInitializeContext
    }
}
