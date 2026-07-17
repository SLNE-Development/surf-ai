package dev.slne.surf.ai.microservice.db.tables

import dev.slne.surf.database.table.AuditableLongIdTable

object AiModelVersionTable : AuditableLongIdTable("ai_model_version") {
    val version = integer("version").uniqueIndex()
    val s3Key = varchar("s3_key", 256)
    val embeddingModelId = varchar("embedding_model_id", 128)
    val metrics = text("metrics") // JSON map<String,Float>
    val trainingDataHash = varchar("training_data_hash", 128)
    val active = bool("active").default(false)
}
