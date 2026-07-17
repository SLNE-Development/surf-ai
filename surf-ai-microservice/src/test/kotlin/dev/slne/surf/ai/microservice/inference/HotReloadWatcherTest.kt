package dev.slne.surf.ai.microservice.inference

import dev.slne.surf.ai.microservice.db.repositories.ModelVersionRepository
import dev.slne.surf.ai.microservice.db.repositories.ModelVersionRow
import dev.slne.surf.ai.microservice.db.tables.AiModelVersionTable
import dev.slne.surf.ai.microservice.storage.ModelStorage
import dev.slne.surf.ai.microservice.testsupport.TestDatabase
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.SchemaUtils
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.deleteAll
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.transactions.suspendTransaction
import kotlinx.coroutines.runBlocking
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

private class CapturingModelHolder : ModelHolder() {
    override val activeVersion: Int = 1
    var reloadedTo: ModelVersionRow? = null

    override suspend fun reloadTo(version: ModelVersionRow, storage: ModelStorage, embeddingPrefix: String) {
        reloadedTo = version
    }
}

class HotReloadWatcherTest {
    @BeforeTest fun setUp() {
        runBlocking {
            TestDatabase.api
            suspendTransaction {
                SchemaUtils.create(AiModelVersionTable)
                AiModelVersionTable.deleteAll()
            }
        }
    }

    @Test fun `tick reloads the holder when the active version changed`() = runBlocking {
        ModelVersionRepository.insertVersion(1, "models/head/head-v1.onnx", "intfloat/multilingual-e5-small", mapOf(), "h1")
        ModelVersionRepository.insertVersion(2, "models/head/head-v2.onnx", "intfloat/multilingual-e5-small", mapOf(), "h2")
        ModelVersionRepository.setActive(2)

        val holder = CapturingModelHolder()
        val storage = ModelStorage("http://localhost:9000", "surf-ai", "surfai", "surfaikey")
        val watcher = HotReloadWatcher(holder, storage, "query: ", pollSeconds = 30)

        watcher.tick()

        assertEquals(2, holder.reloadedTo?.version)
    }
}
