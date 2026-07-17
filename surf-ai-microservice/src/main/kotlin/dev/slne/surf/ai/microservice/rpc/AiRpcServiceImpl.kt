package dev.slne.surf.ai.microservice.rpc

import dev.slne.surf.ai.api.model.AiCategoryScore
import dev.slne.surf.ai.api.model.AiCheckResult
import dev.slne.surf.ai.api.model.AiFeedback
import dev.slne.surf.ai.api.model.AiFeedbackResult
import dev.slne.surf.ai.core.common.rpc.AiRpcService
import dev.slne.surf.ai.microservice.db.repositories.LabeledSampleRepository
import dev.slne.surf.ai.microservice.inference.InferenceService
import dev.slne.surf.ai.microservice.inference.ModelHolder
import dev.slne.surf.ai.microservice.inference.RequestCache
import java.util.UUID

private const val CORRECTION_THRESHOLD = 50f
private const val FALSE_NEGATIVE_SCORE = 95f
private const val FALSE_POSITIVE_SCORE = 0f

class AiRpcServiceImpl(
    private val inferenceService: InferenceService,
    private val requestCache: RequestCache,
    private val modelHolder: ModelHolder,
) : AiRpcService {
    override suspend fun check(input: String): AiCheckResult = inferenceService.check(input)

    override suspend fun feedback(requestId: UUID, feedback: AiFeedback): AiFeedbackResult {
        val entry = requestCache.get(requestId) ?: return AiFeedbackResult.EXPIRED

        val correctedScores = applyFeedback(entry.scores, feedback)
        val labels = correctedScores.filter { it.confidence >= CORRECTION_THRESHOLD }.map { it.category }.toSet()
        val feedbackType = when (feedback) {
            is AiFeedback.FalsePositive -> "FALSE_POSITIVE"
            is AiFeedback.FalseNegative -> "FALSE_NEGATIVE"
            AiFeedback.Correct -> "CORRECT"
        }

        LabeledSampleRepository.insert(entry.text, labels, feedbackType, "ingame", modelHolder.activeVersion)
        inferenceService.applyOverride(entry.text, correctedScores)

        return AiFeedbackResult.ACCEPTED
    }

    private fun applyFeedback(scores: List<AiCategoryScore>, feedback: AiFeedback): List<AiCategoryScore> =
        when (feedback) {
            is AiFeedback.FalsePositive -> {
                val targets = feedback.categories
                    ?: scores.filter { it.confidence >= CORRECTION_THRESHOLD }.map { it.category }.toSet()
                scores.map { if (it.category in targets) it.copy(confidence = FALSE_POSITIVE_SCORE) else it }
            }

            is AiFeedback.FalseNegative ->
                scores.map { if (it.category in feedback.categories) it.copy(confidence = FALSE_NEGATIVE_SCORE) else it }

            AiFeedback.Correct -> scores
        }
}
