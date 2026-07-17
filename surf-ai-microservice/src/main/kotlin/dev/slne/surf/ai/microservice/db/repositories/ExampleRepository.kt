package dev.slne.surf.ai.microservice.db.repositories

import dev.slne.surf.ai.microservice.db.tables.ExampleTable
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.selectAll
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.transactions.suspendTransaction
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.toList

object ExampleRepository {
    suspend fun fetchAllExamples() = suspendTransaction {
        ExampleTable.selectAll()
            .map {
                it[ExampleTable.testUuid]
            }
            .toList()
    }
}