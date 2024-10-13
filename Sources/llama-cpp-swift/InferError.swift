public struct InferError: Error, Sendable {
    public let message: String
    public let code: Code
    
    public enum Code: Int, Sendable {
        case cancelled = 1
        case kvCacheFailure
        case decodingFailure
    }
}
