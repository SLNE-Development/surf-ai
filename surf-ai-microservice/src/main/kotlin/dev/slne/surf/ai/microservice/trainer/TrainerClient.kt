package dev.slne.surf.ai.microservice.trainer

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse

class TrainerClient(private val baseUrl: String) {
    private val client = HttpClient.newHttpClient()

    suspend fun retrain() = withContext(Dispatchers.IO) {
        val request = HttpRequest.newBuilder()
            .uri(URI.create("$baseUrl/retrain"))
            .POST(HttpRequest.BodyPublishers.noBody())
            .build()
        client.send(request, HttpResponse.BodyHandlers.discarding())
        Unit
    }
}
