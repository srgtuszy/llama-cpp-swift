public struct InferError: Error {
    public let message: String
    public let code: Code
    
    public enum Code: Int {
        case cancelled = 1
    }
}
