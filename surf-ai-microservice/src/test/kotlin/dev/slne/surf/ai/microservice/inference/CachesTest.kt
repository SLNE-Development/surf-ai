package dev.slne.surf.ai.microservice.inference

import dev.slne.surf.ai.api.model.AiCategory
import dev.slne.surf.ai.api.model.AiCategoryScore
import java.time.Duration
import java.util.UUID
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class CachesTest {
    @Test fun `RequestCache get returns what was put`() {
        val cache = RequestCache(Duration.ofMinutes(1))
        val id = UUID.randomUUID()
        val entry = RequestCache.Entry("hello", FloatArray(384), listOf(AiCategoryScore(AiCategory.HARASSMENT, 10f)))
        cache.put(id, entry)
        assertEquals(entry, cache.get(id))
    }

    @Test fun `RequestCache entries expire`() {
        val cache = RequestCache(Duration.ofMillis(10))
        val id = UUID.randomUUID()
        cache.put(id, RequestCache.Entry("hello", FloatArray(384), emptyList()))
        Thread.sleep(100)
        assertNull(cache.get(id))
    }

    @Test fun `OverrideCache normalizes text for lookup`() {
        val cache = OverrideCache(100, Duration.ofMinutes(1))
        val scores = listOf(AiCategoryScore(AiCategory.SEXUAL, 95f))
        cache.putCorrection("  HELLO  ", scores)
        assertEquals(scores, cache.get("hello"))
    }

    @Test fun `OverrideCache entries expire`() {
        val cache = OverrideCache(100, Duration.ofMillis(10))
        cache.putCorrection("hello", listOf(AiCategoryScore(AiCategory.SEXUAL, 95f)))
        Thread.sleep(100)
        assertNull(cache.get("hello"))
    }
}
