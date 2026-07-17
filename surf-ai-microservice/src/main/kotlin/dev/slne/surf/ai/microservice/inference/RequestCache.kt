package dev.slne.surf.ai.microservice.inference

import com.github.benmanes.caffeine.cache.Caffeine
import dev.slne.surf.ai.api.model.AiCategoryScore
import java.time.Duration
import java.util.UUID

class RequestCache(ttl: Duration) {
    constructor(ttlMinutes: Long) : this(Duration.ofMinutes(ttlMinutes))

    data class Entry(val text: String, val embedding: FloatArray, val scores: List<AiCategoryScore>)

    private val cache = Caffeine.newBuilder()
        .expireAfterWrite(ttl)
        .build<UUID, Entry>()

    fun put(id: UUID, entry: Entry) = cache.put(id, entry)

    fun get(id: UUID): Entry? = cache.getIfPresent(id)
}
