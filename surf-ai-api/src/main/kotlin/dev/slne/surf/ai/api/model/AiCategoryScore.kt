package dev.slne.surf.ai.api.model

import kotlinx.serialization.Serializable

@Serializable
data class AiCategoryScore(val category: AiCategory, val confidence: Float)
