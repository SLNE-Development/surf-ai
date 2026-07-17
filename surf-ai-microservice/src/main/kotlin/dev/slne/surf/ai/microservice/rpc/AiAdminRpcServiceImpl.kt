package dev.slne.surf.ai.microservice.rpc

import dev.slne.surf.ai.api.model.ModelVersionInfo
import dev.slne.surf.ai.core.common.rpc.AiAdminRpcService
import dev.slne.surf.ai.microservice.db.repositories.LabeledSampleRepository
import dev.slne.surf.ai.microservice.db.repositories.ModelVersionRepository
import dev.slne.surf.ai.microservice.inference.ModelHolder
import dev.slne.surf.ai.microservice.storage.ModelStorage
import dev.slne.surf.ai.microservice.trainer.TrainerClient

class AiAdminRpcServiceImpl(
    private val modelHolder: ModelHolder,
    private val modelStorage: ModelStorage,
    private val embeddingPrefix: String,
    private val trainerClient: TrainerClient,
) : AiAdminRpcService {
    override suspend fun listVersions(): List<ModelVersionInfo> =
        ModelVersionRepository.all().map {
            ModelVersionInfo(
                version = it.version,
                active = it.active,
                createdAtEpochMs = it.createdAtEpochMs,
                metrics = it.metrics,
            )
        }

    override suspend fun rollbackTo(version: Int) {
        ModelVersionRepository.setActive(version)
        val row = ModelVersionRepository.get(version) ?: return
        modelHolder.reloadTo(row, modelStorage, embeddingPrefix)
        LabeledSampleRepository.quarantineAbove(version)
    }

    override suspend fun triggerRetrain() {
        trainerClient.retrain()
    }
}
