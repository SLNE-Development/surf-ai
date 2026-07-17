package dev.slne.surf.ai.client.common

import dev.slne.surf.ai.api.AIInstance

interface AiClientInstance : AIInstance {
    companion object : AiClientInstance by AIInstance.INSTANCE as AiClientInstance {
        val INSTANCE get() = AIInstance.INSTANCE as AiClientInstance
    }
}