package dev.slne.surf.ai.client.common

import dev.slne.surf.ai.core.common.ExampleRpcService

val rabbitApi get() = AiClientInstance.INSTANCE.rabbitApi

val exampleProxy by lazy {
    rabbitApi.createRpcService<ExampleRpcService>()
}