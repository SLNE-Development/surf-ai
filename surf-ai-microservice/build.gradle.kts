import dev.slne.surf.microservice.gradle.plugin.rabbit.RabbitModule

plugins {
    id("dev.slne.surf.api.gradle.standalone")
    id("dev.slne.surf.microservice")
}

surfStandaloneApi {
    withSurfDatabaseR2dbc("2.3.0", "dev.slne.surf.ai.microservice.libs.db")
}

surfMicroservice {
    withRabbitModule(RabbitModule.SERVER_API, true)
    withMicroserviceApi()
}

dependencies {
    api(projects.surfAiCore.surfAiCoreCommon)
    implementation("io.minio:minio:8.5.12")
    implementation("ai.djl.huggingface:tokenizers:0.30.0")
    implementation("com.microsoft.onnxruntime:onnxruntime:1.19.2")
    testImplementation(kotlin("test"))
}