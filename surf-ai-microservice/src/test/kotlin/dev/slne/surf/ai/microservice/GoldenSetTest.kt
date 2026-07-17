package dev.slne.surf.ai.microservice

import dev.slne.surf.ai.api.model.AiCategory
import dev.slne.surf.ai.microservice.db.repositories.ModelVersionRepository
import dev.slne.surf.ai.microservice.inference.BatchingEmbedder
import dev.slne.surf.ai.microservice.inference.Embedder
import dev.slne.surf.ai.microservice.inference.InferenceService
import dev.slne.surf.ai.microservice.inference.ModelHolder
import dev.slne.surf.ai.microservice.inference.OverrideCache
import dev.slne.surf.ai.microservice.inference.RequestCache
import dev.slne.surf.ai.microservice.storage.ModelStorage
import dev.slne.surf.ai.microservice.testsupport.TestDatabase
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import java.time.Duration
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Exercises InferenceService.check with the real, trainer-produced head + embedding model
 * downloaded from the dev MinIO. Needs docker/dev services up and an active model version
 * (run the trainer's bootstrap: `uvicorn app.main:app` then `POST /retrain`, or just start the
 * trainer once - see surf-ai-trainer/README.md). Skipped automatically when there is no active
 * version yet.
 */
class GoldenSetTest {
    private lateinit var inferenceService: InferenceService

    @BeforeTest fun setUp() {
        runBlocking {
            TestDatabase.api
            val activeVersion = ModelVersionRepository.activeVersion()
            assumeTrue("no active model version - run the trainer bootstrap first", activeVersion != null)

            val modelHolder = ModelHolder()
            val storage = ModelStorage("http://localhost:9000", "surf-ai", "surfai", "surfaikey")
            modelHolder.reloadTo(activeVersion!!, storage, "query: ")

            val embedder = Embedder { texts -> modelHolder.embedder()!!.embed(texts) }
            val batchingEmbedder = BatchingEmbedder(embedder, batchWindowMillis = 20, batchMaxSize = 32)
            inferenceService = InferenceService(
                modelHolder,
                batchingEmbedder,
                RequestCache(Duration.ofMinutes(1)),
                OverrideCache(100, Duration.ofMinutes(1)),
            )
        }
    }

    private fun confidenceOf(scores: List<dev.slne.surf.ai.api.model.AiCategoryScore>, category: AiCategory): Float =
        scores.first { it.category == category }.confidence

    @Test fun `kys scores high on SELF_HARM`() = runBlocking {
        val result = inferenceService.check("kys")
        val score = confidenceOf(result.scores, AiCategory.SELF_HARM)
        assertTrue(score > 60f, "expected SELF_HARM > 60, got $score (all: ${result.scores})")
    }

    @Test fun `geh dich doch einfach umbringen scores high on SELF_HARM`() = runBlocking {
        val result = inferenceService.check("geh dich doch einfach umbringen")
        val score = confidenceOf(result.scores, AiCategory.SELF_HARM)
        assertTrue(score > 60f, "expected SELF_HARM > 60, got $score (all: ${result.scores})")
    }

    @Test fun `du hurensohn scores high on HARASSMENT`() = runBlocking {
        val result = inferenceService.check("du hurensohn")
        val score = confidenceOf(result.scores, AiCategory.HARASSMENT)
        assertTrue(score > 60f, "expected HARASSMENT > 60, got $score (all: ${result.scores})")
    }

    @Test fun `penis scores high on SEXUAL`() = runBlocking {
        val result = inferenceService.check("penis")
        val score = confidenceOf(result.scores, AiCategory.SEXUAL)
        assertTrue(score > 40f, "expected SEXUAL > 40, got $score (all: ${result.scores})")
    }

    @Test fun `minecraft hard negative scores low on every category`() = runBlocking {
        val result = inferenceService.check("geh doch einfach hinten auf den berg und spring runter")
        for (score in result.scores) {
            assertTrue(score.confidence < 40f, "expected every score < 40, got ${score.category}=${score.confidence} (all: ${result.scores})")
        }
    }
}
