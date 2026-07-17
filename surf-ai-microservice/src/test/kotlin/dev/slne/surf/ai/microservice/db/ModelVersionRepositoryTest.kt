package dev.slne.surf.ai.microservice.db

import dev.slne.surf.ai.microservice.db.repositories.ModelVersionRepository
import dev.slne.surf.ai.microservice.db.tables.AiModelVersionTable
import dev.slne.surf.ai.microservice.testsupport.TestDatabase
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.SchemaUtils
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.deleteAll
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.transactions.suspendTransaction
import kotlinx.coroutines.runBlocking
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class ModelVersionRepositoryTest {
    @BeforeTest fun setUp() {
        runBlocking {
            TestDatabase.api
            suspendTransaction {
                SchemaUtils.create(AiModelVersionTable)
                AiModelVersionTable.deleteAll()
            }
        }
    }

    @Test fun `setActive flips the active pointer`() = runBlocking {
        ModelVersionRepository.insertVersion(1, "models/head/head-v1.onnx", "intfloat/multilingual-e5-small", mapOf(), "h1")
        ModelVersionRepository.insertVersion(2, "models/head/head-v2.onnx", "intfloat/multilingual-e5-small", mapOf(), "h2")
        ModelVersionRepository.setActive(2)
        assertEquals(2, ModelVersionRepository.activeVersion()!!.version)
    }
}
