package dev.slne.surf.ai.microservice.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.file.Path
import kotlin.math.exp

class ClassificationHead(onnxPath: Path) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = environment.createSession(onnxPath.toString())

    fun score(embeddings: Array<FloatArray>): Array<FloatArray> {
        val batch = embeddings.size
        val dims = embeddings[0].size
        val buffer = FloatBuffer.allocate(batch * dims)
        for (row in embeddings) buffer.put(row)
        buffer.flip()

        OnnxTensor.createTensor(environment, buffer, longArrayOf(batch.toLong(), dims.toLong())).use { tensor ->
            session.run(mapOf("embedding" to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val logits = result.get(0).value as Array<FloatArray>
                return Array(batch) { i -> FloatArray(logits[i].size) { j -> sigmoidTimes100(logits[i][j]) } }
            }
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        fun sigmoidTimes100(logit: Float): Float = (100f / (1f + exp(-logit)))
    }
}
