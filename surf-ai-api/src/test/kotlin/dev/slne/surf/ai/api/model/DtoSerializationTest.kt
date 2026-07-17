package dev.slne.surf.ai.api.model

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.KSerializer
import kotlinx.serialization.cbor.Cbor
import kotlinx.serialization.decodeFromByteArray
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encodeToByteArray
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.modules.SerializersModule
import kotlinx.serialization.modules.contextual
import java.util.UUID
import kotlin.test.Test
import kotlin.test.assertEquals

private object UUIDSerializer : KSerializer<UUID> {
    override val descriptor: SerialDescriptor = PrimitiveSerialDescriptor("UUID", PrimitiveKind.STRING)
    override fun serialize(encoder: Encoder, value: UUID) = encoder.encodeString(value.toString())
    override fun deserialize(decoder: Decoder): UUID = UUID.fromString(decoder.decodeString())
}

@OptIn(ExperimentalSerializationApi::class)
class DtoSerializationTest {
    private val cbor = Cbor {
        ignoreUnknownKeys = true
        serializersModule = SerializersModule { contextual(UUIDSerializer) }
    }

    @Test fun `category order is the wire contract`() {
        assertEquals(0, AiCategory.HARASSMENT.ordinal)
        assertEquals(5, AiCategory.CHILD_SAFETY.ordinal)
        assertEquals(6, AiCategory.entries.size)
    }

    @Test fun `AiCheckResult round-trips through cbor`() {
        val id = UUID.randomUUID()
        val r = AiCheckResult(id, listOf(AiCategoryScore(AiCategory.HARASSMENT, 57.7f)))
        val bytes = cbor.encodeToByteArray(r)
        assertEquals(r, cbor.decodeFromByteArray<AiCheckResult>(bytes))
    }

    @Test fun `feedback polymorphism round-trips`() {
        val fp: AiFeedback = AiFeedback.FalseNegative(setOf(AiCategory.SEXUAL))
        val bytes = cbor.encodeToByteArray(AiFeedback.serializer(), fp)
        assertEquals(fp, cbor.decodeFromByteArray(AiFeedback.serializer(), bytes))
    }
}
