package dev.slne.surf.ai.microservice

import com.google.auto.service.AutoService
import dev.slne.surf.ai.core.common.ExampleRpcService
import dev.slne.surf.ai.microservice.db.tables.AiLabeledSampleTable
import dev.slne.surf.ai.microservice.db.tables.AiModelVersionTable
import dev.slne.surf.ai.microservice.db.tables.AiSeedSampleTable
import dev.slne.surf.ai.microservice.rpc.ExampleRpcServiceImpl
import dev.slne.surf.database.DatabaseApi
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.SchemaUtils
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.transactions.suspendTransaction
import dev.slne.surf.microservice.api.microservice.Microservice
import dev.slne.surf.microservice.api.microservice.getMicroservice
import dev.slne.surf.rabbitmq.api.ServerRabbitMQApi
import java.nio.file.Path
import kotlin.io.path.Path

@AutoService(Microservice::class)
class AiMicroservice : Microservice() {
    override val dataPath: Path = Path("config")

    val rabbitApi = ServerRabbitMQApi.create("surf-ai", dataPath)
    val databaseApi = DatabaseApi.create(dataPath)

    override suspend fun onBootstrap(args: List<String>) {
        suspendTransaction {
            SchemaUtils.create(AiSeedSampleTable, AiLabeledSampleTable, AiModelVersionTable)
        }

        rabbitApi.registerRpcService<ExampleRpcService>(ExampleRpcServiceImpl())
        rabbitApi.freezeAndConnect()
    }

    override suspend fun onDisable() {

    }
}

val microservice get() = getMicroservice<AiMicroservice>()