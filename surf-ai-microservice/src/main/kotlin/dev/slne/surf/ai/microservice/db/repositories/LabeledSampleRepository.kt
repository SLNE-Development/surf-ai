package dev.slne.surf.ai.microservice.db.repositories

import dev.slne.surf.ai.api.model.AiCategory
import dev.slne.surf.ai.microservice.db.tables.AiLabeledSampleTable
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.core.greater
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.insert
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.transactions.suspendTransaction
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.update
import kotlinx.serialization.json.Json

object LabeledSampleRepository {
    suspend fun insert(
        text: String,
        labels: Set<AiCategory>,
        feedbackType: String,
        source: String,
        originModelVersion: Int?,
    ): Unit = suspendTransaction {
        AiLabeledSampleTable.insert {
            it[AiLabeledSampleTable.text] = text
            it[AiLabeledSampleTable.labels] = Json.encodeToString(labels.map { c -> c.name })
            it[AiLabeledSampleTable.feedbackType] = feedbackType
            it[AiLabeledSampleTable.sampleSource] = source
            it[AiLabeledSampleTable.originModelVersion] = originModelVersion
        }
    }

    suspend fun quarantineAbove(version: Int): Unit = suspendTransaction {
        AiLabeledSampleTable.update({ AiLabeledSampleTable.originModelVersion greater version }) {
            it[quarantined] = true
        }
    }
}
