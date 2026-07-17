package dev.slne.surf.ai.microservice.inference

import kotlinx.coroutines.runBlocking
import java.time.Duration
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

private class StubModelHolder : ModelHolder() {
    override fun ready() = true
    override fun scorer() = Scorer { embeddings -> Array(embeddings.size) { floatArrayOf(10f, 90f, 30f, 70f, 50f, 20f) } }
}

class InferenceServiceTest {
    @Test fun `check returns sorted scores and caches the entry`() = runBlocking {
        val modelHolder = StubModelHolder()
        val batchingEmbedder = BatchingEmbedder(
            embedder = Embedder { texts -> Array(texts.size) { FloatArray(384) } },
            batchWindowMillis = 5,
            batchMaxSize = 8,
        )
        val requestCache = RequestCache(Duration.ofMinutes(1))
        val overrideCache = OverrideCache(100, Duration.ofMinutes(1))
        val service = InferenceService(modelHolder, batchingEmbedder, requestCache, overrideCache)

        val result = service.check("x")

        assertEquals(6, result.scores.size)
        assertTrue(result.scores.zipWithNext().all { (a, b) -> a.confidence >= b.confidence })
        val cached = requestCache.get(result.requestId)
        assertEquals("x", cached?.text)
        assertEquals(result.scores, cached?.scores)
    }
}
