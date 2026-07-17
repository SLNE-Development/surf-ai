package dev.slne.surf.ai.microservice.rpc

import dev.slne.surf.ai.core.common.ExampleRpcService

class ExampleRpcServiceImpl : ExampleRpcService {
    override suspend fun exampleMethod(param: String): String {
        return "Hello, $param!"
    }
}