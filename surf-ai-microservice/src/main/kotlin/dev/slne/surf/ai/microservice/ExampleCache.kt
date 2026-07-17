package dev.slne.surf.ai.microservice

import com.github.benmanes.caffeine.cache.Caffeine
import java.util.*

object ExampleCache {
    val exampleCache = Caffeine.newBuilder()
        .maximumSize(10_000)
        .build<UUID, String>()
}