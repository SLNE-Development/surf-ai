package dev.slne.surf.ai.microservice.inference

import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertTrue

class BatchingEmbedderTest {
    @Test fun `concurrent calls are coalesced into few batch invocations`() = runBlocking {
        val batchCallCount = AtomicInteger(0)
        val spy = Embedder { texts ->
            batchCallCount.incrementAndGet()
            Array(texts.size) { FloatArray(384) }
        }
        val batching = BatchingEmbedder(spy, batchWindowMillis = 50, batchMaxSize = 16)

        val results = (1..50).map { i ->
            async {
                delay((i % 5).toLong())
                batching.embed("text-$i")
            }
        }.awaitAll()

        assertTrue(results.all { it.size == 384 })
        assertTrue(batchCallCount.get() < 50, "expected fewer than 50 batch calls, got ${batchCallCount.get()}")
    }
}
