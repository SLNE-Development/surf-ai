package dev.slne.surf.ai.microservice.config

import dev.slne.surf.ai.api.AIInstance
import dev.slne.surf.api.core.config.SpongeYmlConfigClass
import org.spongepowered.configurate.objectmapping.ConfigSerializable

@ConfigSerializable
data class AiConfig(
    val s3Endpoint: String = "http://localhost:9000",
    val s3Bucket: String = "surf-ai",
    val s3AccessKey: String = "surfai",
    val s3SecretKey: String = "surfaikey",
    val trainerBaseUrl: String = "http://localhost:8000",
    val ttlCacheMinutes: Long = 30,
    val overrideCacheMaxSize: Long = 10_000,
    val batchWindowMillis: Long = 20,
    val batchMaxSize: Int = 32,
    val hotReloadPollSeconds: Long = 30,
    val embeddingPrefix: String = "query: ",
) {
    companion object : SpongeYmlConfigClass<AiConfig>(
        configClass = AiConfig::class.java,
        configFolder = AIInstance.dataPath,
        fileName = "config.yml"
    )
}
