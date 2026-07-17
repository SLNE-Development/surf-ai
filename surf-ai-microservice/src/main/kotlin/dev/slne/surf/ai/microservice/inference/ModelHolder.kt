package dev.slne.surf.ai.microservice.inference

import dev.slne.surf.ai.microservice.config.AiConfig
import dev.slne.surf.ai.microservice.db.repositories.ModelVersionRow
import dev.slne.surf.ai.microservice.storage.ModelStorage
import kotlin.io.path.Path
import kotlin.io.path.exists

open class ModelHolder {
    @Volatile private var snapshot: Snapshot? = null

    open val activeVersion: Int? get() = snapshot?.version

    open fun ready(): Boolean = snapshot != null
    open fun embedder(): Embedder? = snapshot?.embedding
    open fun scorer(): Scorer? = snapshot?.head

    open suspend fun reloadTo(version: ModelVersionRow, storage: ModelStorage, config: AiConfig) {
        val embeddingOnnx = Path("cache/models/embedding/model.onnx")
        val tokenizerJson = Path("cache/models/embedding/tokenizer.json")
        if (!embeddingOnnx.exists()) storage.download("models/embedding/model.onnx", embeddingOnnx)
        if (!tokenizerJson.exists()) storage.download("models/embedding/tokenizer.json", tokenizerJson)

        val headOnnx = Path("cache/models/head/head-v${version.version}.onnx")
        storage.download(version.s3Key, headOnnx)

        val embedding = EmbeddingModel(embeddingOnnx, tokenizerJson, config.embeddingPrefix)
        val head = ClassificationHead(headOnnx)

        val previous = snapshot
        snapshot = Snapshot(version.version, embedding, head)
        previous?.let {
            it.embedding.close()
            it.head.close()
        }
    }

    private data class Snapshot(val version: Int, val embedding: EmbeddingModel, val head: ClassificationHead)
}
