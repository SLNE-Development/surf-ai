package dev.slne.surf.ai.api.model

import kotlinx.serialization.Serializable

@Serializable
sealed interface AiFeedback {
    @Serializable data class FalsePositive(val categories: Set<AiCategory>? = null) : AiFeedback
    @Serializable data class FalseNegative(val categories: Set<AiCategory>) : AiFeedback
    @Serializable data object Correct : AiFeedback
}
