plugins {
    id("dev.slne.surf.api.gradle.core")
}

dependencies {
    testImplementation(kotlin("test"))
    testImplementation("org.jetbrains.kotlinx:kotlinx-serialization-cbor:1.11.0")
}