package dev.slne.surf.ai.api.model

import kotlinx.serialization.Serializable

@Serializable
data class ModelVersionInfo(
    val version: Int,
    val active: Boolean,
    val createdAtEpochMs: Long,
    val metrics: Map<String, Float>,
)
