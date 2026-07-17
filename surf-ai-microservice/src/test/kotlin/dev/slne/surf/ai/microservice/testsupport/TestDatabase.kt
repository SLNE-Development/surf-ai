package dev.slne.surf.ai.microservice.testsupport

import dev.slne.surf.database.DatabaseApi
import dev.slne.surf.database.libs.io.r2dbc.postgresql.PostgresqlConnectionConfiguration
import dev.slne.surf.database.libs.io.r2dbc.postgresql.PostgresqlConnectionFactory
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.core.vendors.PostgreSQLDialect
import net.kyori.adventure.text.logger.slf4j.ComponentLogger
import org.slf4j.event.Level

/**
 * Connects directly to the dev-compose Postgres (docker/dev/docker-compose.yml) via the
 * low-level DatabaseApi.create(ConnectionFactory, DatabaseDialect, ...) overload, bypassing the
 * Sponge-config-file bootstrap that DatabaseApi.create(Path) requires (unavailable in bare
 * JUnit tests - it needs a running SurfConfigApi platform context).
 */
object TestDatabase {
    val api: DatabaseApi by lazy {
        val config = PostgresqlConnectionConfiguration.builder()
            .host("localhost")
            .port(5432)
            .username("surf_ai")
            .password("surf_ai")
            .database("surf_ai")
            .build()
        DatabaseApi.create(
            PostgresqlConnectionFactory(config),
            PostgreSQLDialect("PostgreSQL"),
            ComponentLogger.logger(TestDatabase::class.java),
            Level.WARN,
        ) {}
    }
}
