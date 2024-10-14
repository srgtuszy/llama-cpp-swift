import Foundation
import Logging
@preconcurrency import llama

public actor LLama {
  private let logger = Logger.llama
  private let model: Model
  private let sampling: UnsafeMutablePointer<llama_sampler>
  private var tokensList: [llama_token]
  private var temporaryInvalidCChars: [CChar]

  // MARK: - Init & teardown

  public init(model: Model) {
    self.model = model

    // Initialize sampling
    let sparams = llama_sampler_chain_default_params()
    self.sampling = llama_sampler_chain_init(sparams)
    llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.8))
    llama_sampler_chain_add(self.sampling, llama_sampler_init_softmax())
    llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))

    // Initialize token lists
    self.tokensList = []
    self.temporaryInvalidCChars = []
  }

  deinit {
    // llama_sampler_free(sampling)
  }

  // MARK: - Inference

  public func infer(prompt: String, maxTokens: Int32 = 128) -> AsyncThrowingStream<String, Error> {
    return AsyncThrowingStream { continuation in
      Task {
        var isDone = false
        let nLen: Int32 = 1024
        var nCur: Int32 = 0
        var nDecode: Int32 = 0
        var batch = llama_batch_init(512, 0, 1)
        defer {
          llama_batch_free(batch)
        }

        do {
          try self.completionInit(text: prompt, batch: &batch, nLen: nLen, nCur: &nCur)
        } catch {
          continuation.finish(throwing: error)
          return
        }

        while !isDone && nCur < nLen && nCur - batch.n_tokens < maxTokens {
          guard !Task.isCancelled else {
            continuation.finish()
            return
          }
          let newTokenStr = self.completionLoop(
            batch: &batch,
            isDone: &isDone,
            nLen: nLen,
            nCur: &nCur,
            nDecode: &nDecode
          )
          continuation.yield(newTokenStr)
        }
        continuation.finish()
      }
    }
  }

  // MARK: - Private helpers

  private func completionInit(
    text: String,
    batch: inout llama_batch,
    nLen: Int32,
    nCur: inout Int32
  ) throws {
    logger.debug("Attempting to complete \"\(text)\"")

    tokensList = tokenize(text: text, add_bos: true)
    temporaryInvalidCChars = []

    let nCtx = llama_n_ctx(model.context)
    let nKvReq = tokensList.count + Int(nLen) - tokensList.count

    logger.debug("\nn_len = \(nLen), n_ctx = \(nCtx), n_kv_req = \(nKvReq)")

    if nKvReq > nCtx {
      logger.error("Error: n_kv_req > n_ctx, the required KV cache size is not big enough")
      throw InferError(message: "KV cache too small", code: .kvCacheFailure)
    }

    batch.clear()

    for (i, token) in tokensList.enumerated() {
      llamaBatchAdd(&batch, token, Int32(i), [0], false)
    }
    if batch.n_tokens > 0 {
      batch.logits[Int(batch.n_tokens) - 1] = 1  // true
    }

    if llama_decode(model.context, batch) != 0 {
      throw InferError(message: "llama_decode failed", code: .decodingFailure)
    }

    nCur = batch.n_tokens
  }

  private func completionLoop(
    batch: inout llama_batch,
    isDone: inout Bool,
    nLen: Int32,
    nCur: inout Int32,
    nDecode: inout Int32
  ) -> String {
    var newTokenID: llama_token = 0
    newTokenID = llama_sampler_sample(sampling, model.context, batch.n_tokens - 1)

    if llama_token_is_eog(model.model, newTokenID) || nCur == nLen {
      isDone = true
      let newTokenStr = String(
        decoding: Data(temporaryInvalidCChars.map { UInt8(bitPattern: $0) }), as: UTF8.self)
      temporaryInvalidCChars.removeAll()
      return newTokenStr
    }

    let newTokenCChars = tokenToPieceArray(token: newTokenID)
    temporaryInvalidCChars.append(contentsOf: newTokenCChars)
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

    if llama_decode(model.context, batch) != 0 {
      logger.error("Failed to evaluate llama!")
    }

    return newTokenStr
  }

  private func llamaBatchAdd(
    _ batch: inout llama_batch,
    _ id: llama_token,
    _ pos: llama_pos,
    _ seq_ids: [llama_seq_id],
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

  private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
    let utf8Data = text.utf8CString
    let nTokens = Int32(utf8Data.count) + (add_bos ? 1 : 0)
    let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(nTokens))
    defer { tokens.deallocate() }

    let tokenCount = llama_tokenize(
      model.model, text, Int32(utf8Data.count), tokens, Int32(nTokens), add_bos, false)
    guard tokenCount > 0 else {
      return []
    }

    return Array(UnsafeBufferPointer(start: tokens, count: Int(tokenCount)))
  }

  private func tokenToPieceArray(token: llama_token) -> [CChar] {
    var buffer = [CChar](repeating: 0, count: 8)
    var nTokens = llama_token_to_piece(model.model, token, &buffer, 8, 0, false)

    if nTokens < 0 {
      let requiredSize = -nTokens
      buffer = [CChar](repeating: 0, count: Int(requiredSize))
      nTokens = llama_token_to_piece(model.model, token, &buffer, requiredSize, 0, false)
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

extension llama_batch {
  fileprivate mutating func clear() {
    n_tokens = 0
  }
}

extension String {
  fileprivate init?(validatingUTF8 cchars: [CChar]) {
    if #available(macOS 15.0, *) {
      self.init(validating: cchars.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    } else {
      self.init(cString: cchars)
    }
  }
}
