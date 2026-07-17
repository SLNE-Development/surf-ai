package dev.slne.surf.ai.core.common.rpc

import dev.slne.surf.ai.api.model.AiCheckResult
import dev.slne.surf.ai.api.model.AiFeedback
import dev.slne.surf.ai.api.model.AiFeedbackResult
import dev.slne.surf.rabbitmq.api.rpc.RpcService
import kotlinx.serialization.Contextual
import java.util.UUID

@RpcService
interface AiRpcService {
    suspend fun check(input: String): AiCheckResult
    suspend fun feedback(requestId: @Contextual UUID, feedback: AiFeedback): AiFeedbackResult
}
