package dev.slne.surf.ai.microservice.inference

import kotlin.test.Test
import kotlin.test.assertEquals

class EmbeddingModelTest {
    @Test fun `mean pool + l2 normalize`() {
        val hidden = arrayOf(floatArrayOf(1f, 0f), floatArrayOf(3f, 0f)) // 2 tokens, 2 dims
        val mask = longArrayOf(1, 1)
        val v = EmbeddingModel.meanPoolNormalize(hidden, mask)
        assertEquals(1.0f, v[0], 1e-5f) // mean=(2,0)->normalize->(1,0)
        assertEquals(0.0f, v[1], 1e-5f)
    }
}
