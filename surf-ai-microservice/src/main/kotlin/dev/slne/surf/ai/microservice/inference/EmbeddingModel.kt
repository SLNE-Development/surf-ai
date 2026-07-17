package dev.slne.surf.ai.microservice.inference

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer
import java.nio.file.Path
import kotlin.math.sqrt

class EmbeddingModel(onnxPath: Path, tokenizerPath: Path, private val prefix: String) : Embedder, AutoCloseable {
    private val tokenizer = HuggingFaceTokenizer.builder()
        .optTokenizerPath(tokenizerPath)
        .optPadding(true)
        .optTruncation(true)
        .build()

    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = environment.createSession(onnxPath.toString())

    override fun embed(texts: List<String>): Array<FloatArray> {
        val encodings = tokenizer.batchEncode(texts.map { prefix + it })
        val batch = encodings.size
        val seqLen = encodings.maxOf { it.ids.size }

        val idsBuffer = LongBuffer.allocate(batch * seqLen)
        val maskBuffer = LongBuffer.allocate(batch * seqLen)
        val tokenTypeBuffer = LongBuffer.allocate(batch * seqLen)
        val masks = Array(batch) { LongArray(seqLen) }
        for (i in 0 until batch) {
            val ids = encodings[i].ids
            val mask = encodings[i].attentionMask
            for (j in 0 until seqLen) {
                idsBuffer.put(if (j < ids.size) ids[j] else 0L)
                val m = if (j < mask.size) mask[j] else 0L
                maskBuffer.put(m)
                tokenTypeBuffer.put(0L)
                masks[i][j] = m
            }
        }
        idsBuffer.flip()
        maskBuffer.flip()
        tokenTypeBuffer.flip()

        val shape = longArrayOf(batch.toLong(), seqLen.toLong())
        OnnxTensor.createTensor(environment, idsBuffer, shape).use { idsTensor ->
            OnnxTensor.createTensor(environment, maskBuffer, shape).use { maskTensor ->
                OnnxTensor.createTensor(environment, tokenTypeBuffer, shape).use { tokenTypeTensor ->
                    session.run(
                        mapOf(
                            "input_ids" to idsTensor,
                            "attention_mask" to maskTensor,
                            "token_type_ids" to tokenTypeTensor,
                        )
                    ).use { result ->
                        @Suppress("UNCHECKED_CAST")
                        val hiddenState = result.get(0).value as Array<Array<FloatArray>>
                        return Array(batch) { i -> meanPoolNormalize(hiddenState[i], masks[i]) }
                    }
                }
            }
        }
    }

    override fun close() {
        session.close()
        tokenizer.close()
    }

    companion object {
        fun meanPoolNormalize(hidden: Array<FloatArray>, mask: LongArray): FloatArray {
            val dims = hidden[0].size
            val sum = FloatArray(dims)
            var count = 0
            for (t in hidden.indices) {
                if (mask[t] == 1L) {
                    count++
                    for (d in 0 until dims) sum[d] += hidden[t][d]
                }
            }
            if (count > 0) for (d in 0 until dims) sum[d] /= count

            var norm = 0.0
            for (d in 0 until dims) norm += sum[d].toDouble() * sum[d].toDouble()
            norm = sqrt(norm)
            if (norm > 0.0) for (d in 0 until dims) sum[d] = (sum[d] / norm).toFloat()
            return sum
        }
    }
}
