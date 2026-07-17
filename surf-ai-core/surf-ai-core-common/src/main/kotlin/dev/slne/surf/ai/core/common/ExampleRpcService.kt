package dev.slne.surf.ai.core.common

import dev.slne.surf.rabbitmq.api.rpc.RpcService

@RpcService
interface ExampleRpcService {
    suspend fun exampleMethod(param: String): String
}