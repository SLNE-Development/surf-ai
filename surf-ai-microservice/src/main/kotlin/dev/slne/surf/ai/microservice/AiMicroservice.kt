package dev.slne.surf.ai.microservice

import com.google.auto.service.AutoService
import dev.slne.surf.ai.core.common.ExampleRpcService
import dev.slne.surf.ai.core.common.rpc.AiAdminRpcService
import dev.slne.surf.ai.core.common.rpc.AiRpcService
import dev.slne.surf.ai.microservice.config.AiConfig
import dev.slne.surf.ai.microservice.db.tables.AiLabeledSampleTable
import dev.slne.surf.ai.microservice.db.tables.AiModelVersionTable
import dev.slne.surf.ai.microservice.db.tables.AiSeedSampleTable
import dev.slne.surf.ai.microservice.inference.BatchingEmbedder
import dev.slne.surf.ai.microservice.inference.Embedder
import dev.slne.surf.ai.microservice.inference.HotReloadWatcher
import dev.slne.surf.ai.microservice.inference.InferenceService
import dev.slne.surf.ai.microservice.inference.ModelHolder
import dev.slne.surf.ai.microservice.inference.OverrideCache
import dev.slne.surf.ai.microservice.inference.RequestCache
import dev.slne.surf.ai.microservice.rpc.AiAdminRpcServiceImpl
import dev.slne.surf.ai.microservice.rpc.AiRpcServiceImpl
import dev.slne.surf.ai.microservice.rpc.ExampleRpcServiceImpl
import dev.slne.surf.ai.microservice.storage.ModelStorage
import dev.slne.surf.ai.microservice.trainer.TrainerClient
import dev.slne.surf.database.DatabaseApi
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.SchemaUtils
import dev.slne.surf.database.libs.org.jetbrains.exposed.v1.r2dbc.transactions.suspendTransaction
import dev.slne.surf.microservice.api.microservice.Microservice
import dev.slne.surf.microservice.api.microservice.getMicroservice
import dev.slne.surf.rabbitmq.api.ServerRabbitMQApi
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import java.nio.file.Path
import kotlin.io.path.Path

@AutoService(Microservice::class)
class AiMicroservice : Microservice() {
    override val dataPath: Path = Path("config")

    val rabbitApi = ServerRabbitMQApi.create("surf-ai", dataPath)
    val databaseApi = DatabaseApi.create(dataPath)

    val modelHolder = ModelHolder()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    override suspend fun onBootstrap(args: List<String>) {
        suspendTransaction {
            SchemaUtils.create(AiSeedSampleTable, AiLabeledSampleTable, AiModelVersionTable)
        }

        val config = AiConfig.getConfig()
        val modelStorage = ModelStorage(config)
        val embedder = Embedder { texts -> modelHolder.embedder()!!.embed(texts) }
        val batchingEmbedder = BatchingEmbedder(embedder, config.batchWindowMillis, config.batchMaxSize)
        val requestCache = RequestCache(config.ttlCacheMinutes)
        val overrideCache = OverrideCache(config.overrideCacheMaxSize, config.ttlCacheMinutes)
        val inferenceService = InferenceService(modelHolder, batchingEmbedder, requestCache, overrideCache)
        val trainerClient = TrainerClient(config.trainerBaseUrl)

        rabbitApi.registerRpcService<ExampleRpcService>(ExampleRpcServiceImpl())
        rabbitApi.registerRpcService<AiRpcService>(AiRpcServiceImpl(inferenceService, requestCache, modelHolder))
        rabbitApi.registerRpcService<AiAdminRpcService>(
            AiAdminRpcServiceImpl(modelHolder, modelStorage, config.embeddingPrefix, trainerClient)
        )
        rabbitApi.freezeAndConnect()

        HotReloadWatcher(modelHolder, modelStorage, config.embeddingPrefix, config.hotReloadPollSeconds).start(scope)
    }

    override suspend fun onDisable() {

    }
}

val microservice get() = getMicroservice<AiMicroservice>()