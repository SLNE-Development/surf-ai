package dev.slne.surf.ai.microservice.config

import dev.slne.surf.ai.api.AIInstance
import dev.slne.surf.api.core.config.SpongeYmlConfigClass
import org.spongepowered.configurate.objectmapping.ConfigSerializable

@ConfigSerializable
data class AiConfig(
    val something: String
) {
    companion object : SpongeYmlConfigClass<AiConfig>(
        configClass = AiConfig::class.java,
        configFolder = AIInstance.dataPath,
        fileName = "config.yml"
    )
}
