package dev.slne.surf.ai.core.common.rpc

import dev.slne.surf.ai.api.model.ModelVersionInfo
import dev.slne.surf.rabbitmq.api.rpc.RpcService

@RpcService
interface AiAdminRpcService {
    suspend fun listVersions(): List<ModelVersionInfo>
    suspend fun rollbackTo(version: Int)
    suspend fun triggerRetrain()
}
