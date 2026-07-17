package dev.slne.surf.ai.microservice.db.tables

import dev.slne.surf.database.columns.nativeUuid
import dev.slne.surf.database.table.AuditableLongIdTable

// If you need createdAt and updatedAt, use AuditableLongIdTable instead of LongIdTable
object ExampleTable : AuditableLongIdTable("example_table") {
    val testUuid = nativeUuid("test_uuid")
}