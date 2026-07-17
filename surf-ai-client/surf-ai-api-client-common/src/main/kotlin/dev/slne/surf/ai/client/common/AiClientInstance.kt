package dev.slne.surf.ai.client.common

import dev.slne.surf.ai.api.AIInstance
import dev.slne.surf.ai.api.model.AiFeedback
import dev.slne.surf.rabbitmq.api.ClientRabbitMQApi
import java.util.UUID

abstract class AiClientInstance : AIInstance {
    val rabbitApi: ClientRabbitMQApi = ClientRabbitMQApi.create("surf-ai", dataPath)

    override suspend fun check(input: String) = aiProxy.check(input)
    override suspend fun feedback(requestId: UUID, feedback: AiFeedback) = aiProxy.feedback(requestId, feedback)

    final suspend fun onLoad() {
        rabbitApi.freezeAndConnect()
    }

    final suspend fun onEnable() {

    }

    final suspend fun onDisable() {

    }

    companion object {
        val INSTANCE get() = AIInstance.INSTANCE as AiClientInstance
    }
}