package dev.slne.surf.ai.microservice.inference

import com.github.benmanes.caffeine.cache.Caffeine
import dev.slne.surf.ai.api.model.AiCategoryScore
import java.time.Duration

fun normalize(text: String): String = text.trim().lowercase().replace(Regex("\\s+"), " ")

class OverrideCache(maxSize: Long, ttl: Duration) {
    constructor(maxSize: Long, ttlMinutes: Long) : this(maxSize, Duration.ofMinutes(ttlMinutes))

    private val cache = Caffeine.newBuilder()
        .maximumSize(maxSize)
        .expireAfterWrite(ttl)
        .build<String, List<AiCategoryScore>>()

    fun putCorrection(text: String, scores: List<AiCategoryScore>) = cache.put(normalize(text), scores)

    fun get(text: String): List<AiCategoryScore>? = cache.getIfPresent(normalize(text))
}
