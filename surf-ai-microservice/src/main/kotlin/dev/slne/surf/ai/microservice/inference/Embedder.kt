package dev.slne.surf.ai.microservice.inference

fun interface Embedder {
    fun embed(texts: List<String>): Array<FloatArray>
}
