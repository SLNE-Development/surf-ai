package dev.slne.surf.ai.client.common

import dev.slne.surf.ai.core.common.ExampleRpcService
import dev.slne.surf.ai.core.common.rpc.AiAdminRpcService
import dev.slne.surf.ai.core.common.rpc.AiRpcService

val rabbitApi get() = AiClientInstance.INSTANCE.rabbitApi

val exampleProxy by lazy {
    rabbitApi.createRpcService<ExampleRpcService>()
}

val aiProxy by lazy { rabbitApi.createRpcService<AiRpcService>() }
val aiAdminProxy by lazy { rabbitApi.createRpcService<AiAdminRpcService>() }