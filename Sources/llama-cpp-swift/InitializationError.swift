public struct InitializationError: Error, Sendable {
    public let message: String
    public let code: Code
    
    public enum Code: Int, Sendable {
        case failedToLoadModel = 1
        case failedToInitializeContext
    }
}
