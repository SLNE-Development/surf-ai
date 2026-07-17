package dev.slne.surf.ai.api

import dev.slne.surf.api.core.util.requiredService
import java.nio.file.Path

interface AIInstance {
    val dataPath: Path

    companion object : AIInstance by instance {
        val INSTANCE get() = instance
    }
}

private val instance = requiredService<AIInstance>()