package dev.slne.surf.ai.microservice.db.tables

import dev.slne.surf.database.table.AuditableLongIdTable

// only path for live chat into the DB
object AiLabeledSampleTable : AuditableLongIdTable("ai_labeled_sample") {
    val text = text("text")
    val labels = text("labels") // JSON array of AiCategory names (corrected truth)
    val feedbackType = varchar("feedback_type", 32) // FALSE_POSITIVE | FALSE_NEGATIVE | CORRECT
    val sampleSource = varchar("source", 64)
    val quarantined = bool("quarantined").default(false)
    val weight = float("weight").default(1f)
    val originModelVersion = integer("origin_model_version").nullable()
}
