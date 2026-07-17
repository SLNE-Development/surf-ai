package dev.slne.surf.ai.client.velocity

import com.google.auto.service.AutoService
import dev.slne.surf.ai.api.AIInstance
import dev.slne.surf.ai.client.common.AiClientInstance
import java.nio.file.Path

@AutoService(AiClientInstance::class, AIInstance::class)
class AiVelocityClientInstance : AiClientInstance() {
    override val dataPath: Path
        get() = plugin.dataPath
}