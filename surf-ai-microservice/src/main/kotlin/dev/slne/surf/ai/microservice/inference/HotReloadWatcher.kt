package dev.slne.surf.ai.microservice.inference

import dev.slne.surf.ai.microservice.db.repositories.ModelVersionRepository
import dev.slne.surf.ai.microservice.storage.ModelStorage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class HotReloadWatcher(
    private val modelHolder: ModelHolder,
    private val modelStorage: ModelStorage,
    private val embeddingPrefix: String,
    private val pollSeconds: Long,
) {
    suspend fun tick() {
        val active = ModelVersionRepository.activeVersion() ?: return
        if (active.version != modelHolder.activeVersion) {
            modelHolder.reloadTo(active, modelStorage, embeddingPrefix)
        }
    }

    fun start(scope: CoroutineScope) {
        scope.launch {
            while (isActive) {
                tick()
                delay(pollSeconds * 1000)
            }
        }
    }
}
