plugins {
    id("dev.slne.surf.api.gradle.paper-plugin")
}

surfPaperPluginApi {
    mainClass("dev.slne.surf.api.client.paper.PaperMain")
}

dependencies {
    api(projects.surfAiClient.surfAiApiClientCommon)
}