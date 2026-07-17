package dev.slne.surf.ai.api.model

import kotlinx.serialization.Contextual
import kotlinx.serialization.Serializable
import java.util.UUID

@Serializable
data class AiCheckResult(
    @Contextual val requestId: UUID,
    val scores: List<AiCategoryScore>,
)
