pluginManagement {
    repositories {
        gradlePluginPortal()
        maven("https://reposilite.slne.dev/public/")
    }
}

plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
    id("dev.slne.surf.api.gradle.settings") version "+"
}

rootProject.name = "surf-ai"

include("surf-ai-api")
include("surf-ai-client:surf-ai-api-client-common")
include("surf-ai-client:surf-ai-api-client-paper")
include("surf-ai-client:surf-ai-api-client-velocity")
include("surf-ai-microservice")
include("surf-ai-core:surf-ai-core-common")
include("surf-ai-core:surf-ai-core-client")