package dev.slne.surf.ai.microservice.db.tables

import dev.slne.surf.database.table.AuditableLongIdTable

object AiSeedSampleTable : AuditableLongIdTable("ai_seed_sample") {
    val text = text("text")
    val labels = text("labels") // JSON array of AiCategory names
    val language = varchar("language", 8)
    val sampleSource = varchar("source", 64)
}
