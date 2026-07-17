package dev.slne.surf.ai.microservice.inference

fun interface Scorer {
    fun score(embeddings: Array<FloatArray>): Array<FloatArray>
}
