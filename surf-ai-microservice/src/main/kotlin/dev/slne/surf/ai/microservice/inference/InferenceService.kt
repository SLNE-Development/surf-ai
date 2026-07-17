package dev.slne.surf.ai.microservice.inference

import dev.slne.surf.ai.api.model.AiCategory
import dev.slne.surf.ai.api.model.AiCategoryScore
import dev.slne.surf.ai.api.model.AiCheckResult
import java.util.UUID

class InferenceService(
    private val modelHolder: ModelHolder,
    private val batchingEmbedder: BatchingEmbedder,
    private val requestCache: RequestCache,
    private val overrideCache: OverrideCache,
) {
    suspend fun check(text: String): AiCheckResult {
        if (!modelHolder.ready()) return AiCheckResult(UUID.randomUUID(), emptyList())

        val embedding = batchingEmbedder.embed(text)
        val override = overrideCache.get(text)
        val scores = override ?: run {
            val logitsRow = modelHolder.scorer()!!.score(arrayOf(embedding))[0]
            AiCategory.entries.mapIndexed { i, category -> AiCategoryScore(category, logitsRow[i]) }
        }
        val sorted = scores.sortedByDescending { it.confidence }

        val requestId = UUID.randomUUID()
        requestCache.put(requestId, RequestCache.Entry(text, embedding, sorted))
        return AiCheckResult(requestId, sorted)
    }

    fun applyOverride(text: String, corrected: List<AiCategoryScore>) {
        overrideCache.putCorrection(text, corrected)
    }
}
