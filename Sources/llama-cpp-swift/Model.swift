import Foundation
import Logging
import llama

public final class Model {
  private let logger = Logger.llama
  let model: OpaquePointer
  let context: OpaquePointer

  public init(modelPath: String, contextSize: UInt32 = 2048) throws {
    llama_backend_init()
    var modelParams = llama_model_default_params()

    #if targetEnvironment(simulator)
      modelParams.n_gpu_layers = 0
      logger.debug("Running on simulator, force use n_gpu_layers = 0")
    #endif

    guard let model = llama_load_model_from_file(modelPath, modelParams) else {
      llama_backend_free()
      throw InitializationError(message: "Failed to load model", code: .failedToLoadModel)
    }
    self.model = model

    // Initialize context parameters
    let nThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
    logger.debug("Using \(nThreads) threads")

    var ctxParams = llama_context_default_params()
    ctxParams.n_ctx = contextSize
    ctxParams.n_threads = Int32(nThreads)
    ctxParams.n_threads_batch = Int32(nThreads)

    guard let context = llama_new_context_with_model(model, ctxParams) else {
      llama_free_model(model)
      llama_backend_free()
      throw InitializationError(
        message: "Failed to initialize context", code: .failedToInitializeContext)
    }
    self.context = context
  }

  deinit {
    llama_free(context)
    llama_free_model(model)
    llama_backend_free()
  }
}
