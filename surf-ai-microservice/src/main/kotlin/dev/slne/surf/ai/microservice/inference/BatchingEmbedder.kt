package dev.slne.surf.ai.microservice.inference

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.selects.onTimeout
import kotlinx.coroutines.selects.select

@OptIn(ExperimentalCoroutinesApi::class)
class BatchingEmbedder(
    private val embedder: Embedder,
    private val batchWindowMillis: Long,
    private val batchMaxSize: Int,
    scope: CoroutineScope = CoroutineScope(Dispatchers.Default),
) {
    private data class Request(val text: String, val result: CompletableDeferred<FloatArray>)

    private val channel = Channel<Request>(Channel.UNLIMITED)

    init {
        scope.launch { consume() }
    }

    suspend fun embed(text: String): FloatArray {
        val deferred = CompletableDeferred<FloatArray>()
        channel.send(Request(text, deferred))
        return deferred.await()
    }

    private suspend fun consume() {
        while (true) {
            val batch = ArrayList<Request>(batchMaxSize)
            batch += channel.receive()
            val deadline = System.currentTimeMillis() + batchWindowMillis
            while (batch.size < batchMaxSize) {
                val remaining = deadline - System.currentTimeMillis()
                if (remaining <= 0) break
                val next = select {
                    channel.onReceive { it }
                    onTimeout(remaining) { null }
                }
                if (next == null) break
                batch += next
            }

            val results = embedder.embed(batch.map { it.text })
            for (i in batch.indices) batch[i].result.complete(results[i])
        }
    }
}
