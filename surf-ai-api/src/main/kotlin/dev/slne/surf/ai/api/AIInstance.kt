package dev.slne.surf.ai.api

import dev.slne.surf.ai.api.model.AiCheckResult
import dev.slne.surf.ai.api.model.AiFeedback
import dev.slne.surf.ai.api.model.AiFeedbackResult
import dev.slne.surf.api.core.util.requiredService
import java.nio.file.Path
import java.util.UUID

interface AIInstance {
    val dataPath: Path

    suspend fun check(input: String): AiCheckResult
    suspend fun feedback(requestId: UUID, feedback: AiFeedback): AiFeedbackResult

    companion object : AIInstance by instance {
        val INSTANCE get() = instance
    }
}

private val instance = requiredService<AIInstance>()