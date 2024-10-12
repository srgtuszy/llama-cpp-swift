import Foundation
import Logging
import llama

public actor LLama {
  private let logger = Logger.llama
  private let model: OpaquePointer
  private let context: OpaquePointer
  private let sampling: UnsafeMutablePointer<llama_sampler>
  private var batch: llama_batch
  private var tokensList: [llama_token]
  private var temporaryInvalidCChars: [CChar]
  private var isDone = false

  private var nLen: Int32 = 1024
  private var nCur: Int32 = 0
  private var nDecode: Int32 = 0

  // MARK: - Init & teardown

  public init(modelPath: String, contextSize: UInt32 = 2048) throws {
    llama_backend_init()
    let modelParams = llama_model_default_params()

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

    // Initialize sampling
    let sparams = llama_sampler_chain_default_params()
    self.sampling = llama_sampler_chain_init(sparams)
    llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.8))
    llama_sampler_chain_add(self.sampling, llama_sampler_init_softmax())
    llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))

    // Initialize batch and token list
    self.batch = llama_batch_init(512, 0, 1)
    self.tokensList = []
    self.temporaryInvalidCChars = []
  }

  deinit {
    llama_sampler_free(sampling)
    llama_batch_free(batch)
    llama_free(context)
    llama_free_model(model)
    llama_backend_free()
  }

  // MARK: - Inference

  public func infer(prompt: String, maxTokens: Int32 = 128) async throws -> String {
    completionInit(text: prompt)
    var generatedText = ""

    while !isDone && nCur < nLen && nCur - batch.n_tokens < maxTokens {
      guard !Task.isCancelled else {
        throw InferError(message: "Task cancelled", code: .cancelled)
      }
      let newTokenStr = completionLoop()
      generatedText += newTokenStr
    }

    return generatedText
  }

  // MARK: - Private helpers

  private func llamaBatchClear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
  }

  private func llamaBatchAdd(
    _ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id],
    _ logits: Bool
  ) {
    batch.token[Int(batch.n_tokens)] = id
    batch.pos[Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
      batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits[Int(batch.n_tokens)] = logits ? 1 : 0

    batch.n_tokens += 1
  }

  private func completionInit(text: String) {
    logger.debug("Attempting to complete \"\(text)\"")

    tokensList = tokenize(text: text, add_bos: true)
    temporaryInvalidCChars = []

    let nCtx = llama_n_ctx(context)
    let nKvReq = tokensList.count + Int(nLen) - tokensList.count

    logger.debug("\nn_len = \(self.nLen), n_ctx = \(nCtx), n_kv_req = \(nKvReq)")

    if nKvReq > nCtx {
      print("Error: n_kv_req > n_ctx, the required KV cache size is not big enough")
    }

    batch.clear()

    for (i, token) in tokensList.enumerated() {
      llamaBatchAdd(&batch, token, Int32(i), [0], false)
    }
    if batch.n_tokens > 0 {
      batch.logits[Int(batch.n_tokens) - 1] = 1  // true
    }

    if llama_decode(context, batch) != 0 {
      print("llama_decode() failed")
    }

    nCur = batch.n_tokens
  }

  private func completionLoop() -> String {
    var newTokenID: llama_token = 0
    newTokenID = llama_sampler_sample(sampling, context, batch.n_tokens - 1)

    if llama_token_is_eog(model, newTokenID) || nCur == nLen {
      isDone = true
      let newTokenStr = String(cString: temporaryInvalidCChars + [0])
      temporaryInvalidCChars.removeAll()
      return newTokenStr
    }

    let newTokenCChars = tokenToPieceArray(token: newTokenID)
    temporaryInvalidCChars.append(contentsOf: newTokenCChars + [0])
    let newTokenStr: String

    if let string = String(validatingUTF8: temporaryInvalidCChars) {
      temporaryInvalidCChars.removeAll()
      newTokenStr = string
    } else if let partialStr = attemptPartialString(from: temporaryInvalidCChars) {
      temporaryInvalidCChars.removeAll()
      newTokenStr = partialStr
    } else {
      newTokenStr = ""
    }

    batch.clear()
    llamaBatchAdd(&batch, newTokenID, nCur, [0], true)

    nDecode += 1
    nCur += 1

    if llama_decode(context, batch) != 0 {
      print("Failed to evaluate llama!")
    }

    return newTokenStr
  }

  private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
    let utf8Data = text.utf8CString
    let nTokens = Int32(utf8Data.count) + (add_bos ? 1 : 0)
    let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(nTokens))
    defer { tokens.deallocate() }

    let tokenCount = llama_tokenize(
      model, text, Int32(utf8Data.count), tokens, Int32(nTokens), add_bos, false)
    guard tokenCount > 0 else {
      return []
    }

    return Array(UnsafeBufferPointer(start: tokens, count: Int(tokenCount)))
  }

  private func tokenToPiece(token: llama_token) -> String? {
    var result = [CChar](repeating: 0, count: 8)
    var nTokens = llama_token_to_piece(model, token, &result, 8, 0, false)

    if nTokens < 0 {
      let requiredSize = -nTokens
      result = [CChar](repeating: 0, count: Int(requiredSize))
      nTokens = llama_token_to_piece(model, token, &result, requiredSize, 0, false)
    }

    return String(cString: result)
  }

  private func tokenToPieceArray(token: llama_token) -> [CChar] {
    var buffer = [CChar](repeating: 0, count: 8)
    var nTokens = llama_token_to_piece(model, token, &buffer, 8, 0, false)

    if nTokens < 0 {
      let requiredSize = -nTokens
      buffer = [CChar](repeating: 0, count: Int(requiredSize))
      nTokens = llama_token_to_piece(model, token, &buffer, requiredSize, 0, false)
    }

    return Array(buffer.prefix(Int(nTokens)))
  }

  private func attemptPartialString(from cchars: [CChar]) -> String? {
    for i in (1..<cchars.count).reversed() {
      let subArray = Array(cchars.prefix(i))
      if let str = String(validatingUTF8: subArray) {
        return str
      }
    }
    return nil
  }
}

private extension llama_batch {
  mutating func clear() {
    n_tokens = 0
  }
}
