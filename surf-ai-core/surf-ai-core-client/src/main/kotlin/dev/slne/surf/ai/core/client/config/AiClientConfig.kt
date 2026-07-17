package dev.slne.surf.ai.core.client.config

import dev.slne.surf.ai.api.AIInstance
import dev.slne.surf.api.core.config.SpongeYmlConfigClass
import org.spongepowered.configurate.objectmapping.ConfigSerializable

@ConfigSerializable
data class AiClientConfig(
    val something: String
) {
    companion object : SpongeYmlConfigClass<AiClientConfig>(
        configClass = AiClientConfig::class.java,
        configFolder = AIInstance.dataPath,
        fileName = "config.yml"
    )
}
