package dev.slne.surf.ai.api.model

import kotlinx.serialization.Serializable

@Serializable
enum class AiCategory { HARASSMENT, SELF_HARM, HATE_SPEECH, SEXUAL, THREAT, CHILD_SAFETY }
