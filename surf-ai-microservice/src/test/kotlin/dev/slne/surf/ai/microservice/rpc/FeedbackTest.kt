package dev.slne.surf.ai.microservice.rpc

import dev.slne.surf.ai.api.model.AiCategory
import dev.slne.surf.ai.api.model.AiCategoryScore
import dev.slne.surf.ai.api.model.AiFeedback
import dev.slne.surf.ai.api.model.AiFeedbackResult
import dev.slne.surf.ai.microservice.db.tables.AiLabeledSampleTable
import dev.slne.surf.ai.microservice.inference.BatchingEmbedder
import dev.slne.surf.ai.microservice.inference.Embedder
import dev.slne.surf.ai.microservice.inference.InferenceService
import dev.slne.surf.ai.microservice.inference.ModelHolder
import dev.slne.surf.ai.microservice.inference.OverrideCache
import dev.slne.surf.ai.microservice.inference.RequestCache
import dev.slne.surf.ai.microservice.testsupport.TestDatabase
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.SchemaUtils
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.deleteAll
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.selectAll
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.transactions.suspendTransaction
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import java.time.Duration
import java.util.UUID
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FeedbackTest {
    private val requestCache = RequestCache(Duration.ofMinutes(1))
    private val overrideCache = OverrideCache(100, Duration.ofMinutes(1))
    private val modelHolder = ModelHolder()
    private val inferenceService = InferenceService(
        modelHolder,
        BatchingEmbedder(Embedder { texts -> Array(texts.size) { FloatArray(384) } }, 5, 8),
        requestCache,
        overrideCache,
    )
    private val service = AiRpcServiceImpl(inferenceService, requestCache, modelHolder)

    @BeforeTest fun setUp() {
        runBlocking {
            TestDatabase.api
            suspendTransaction {
                SchemaUtils.create(AiLabeledSampleTable)
                AiLabeledSampleTable.deleteAll()
            }
        }
    }

    @Test fun `unknown request id expires`() = runBlocking {
        assertEquals(AiFeedbackResult.EXPIRED, service.feedback(UUID.randomUUID(), AiFeedback.Correct))
    }

    @Test fun `false negative persists label and applies override`() = runBlocking {
        val id = UUID.randomUUID()
        val text = "some chat message"
        val initialScores = AiCategory.entries.map { AiCategoryScore(it, 10f) }
        requestCache.put(id, RequestCache.Entry(text, FloatArray(384), initialScores))

        val result = service.feedback(id, AiFeedback.FalseNegative(setOf(AiCategory.SEXUAL)))

        assertEquals(AiFeedbackResult.ACCEPTED, result)

        val rows = suspendTransaction {
            AiLabeledSampleTable.selectAll().toList()
        }
        assertEquals(1, rows.size)
        assertTrue(rows[0][AiLabeledSampleTable.labels].contains("SEXUAL"))

        val corrected = overrideCache.get(text)
        assertEquals(95f, corrected?.first { it.category == AiCategory.SEXUAL }?.confidence)
    }
}
