package dev.slne.surf.ai.microservice.db.repositories

import dev.slne.surf.ai.microservice.db.tables.AiModelVersionTable
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.core.ResultRow
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.core.eq
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.insert
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.selectAll
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.transactions.suspendTransaction
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.update
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.toList
import kotlinx.serialization.json.Json

data class ModelVersionRow(
    val version: Int,
    val s3Key: String,
    val metrics: Map<String, Float>,
    val active: Boolean,
    val createdAtEpochMs: Long,
)

object ModelVersionRepository {
    suspend fun insertVersion(
        version: Int,
        s3Key: String,
        embeddingModelId: String,
        metrics: Map<String, Float>,
        trainingDataHash: String,
    ): Unit = suspendTransaction {
        AiModelVersionTable.insert {
            it[AiModelVersionTable.version] = version
            it[AiModelVersionTable.s3Key] = s3Key
            it[AiModelVersionTable.embeddingModelId] = embeddingModelId
            it[AiModelVersionTable.metrics] = Json.encodeToString(metrics)
            it[AiModelVersionTable.trainingDataHash] = trainingDataHash
        }
    }

    suspend fun setActive(version: Int): Unit = suspendTransaction {
        AiModelVersionTable.update { it[active] = false }
        AiModelVersionTable.update({ AiModelVersionTable.version eq version }) { it[active] = true }
    }

    suspend fun activeVersion(): ModelVersionRow? = suspendTransaction {
        AiModelVersionTable.selectAll()
            .where { AiModelVersionTable.active eq true }
            .map(::toRow)
            .firstOrNull()
    }

    suspend fun all(): List<ModelVersionRow> = suspendTransaction {
        AiModelVersionTable.selectAll().map(::toRow).toList()
    }

    private fun toRow(row: ResultRow) = ModelVersionRow(
        version = row[AiModelVersionTable.version],
        s3Key = row[AiModelVersionTable.s3Key],
        metrics = Json.decodeFromString(row[AiModelVersionTable.metrics]),
        active = row[AiModelVersionTable.active],
        createdAtEpochMs = row[AiModelVersionTable.createdAt].toInstant().toEpochMilli(),
    )
}
