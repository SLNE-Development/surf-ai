package dev.slne.surf.ai.client.common

import dev.slne.surf.ai.api.AIInstance
import dev.slne.surf.rabbitmq.api.ClientRabbitMQApi

abstract class AiClientInstance : AIInstance {
    val rabbitApi: ClientRabbitMQApi = ClientRabbitMQApi.create("surf-ai", dataPath)

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