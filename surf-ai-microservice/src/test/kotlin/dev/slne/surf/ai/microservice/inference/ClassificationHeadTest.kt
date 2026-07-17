package dev.slne.surf.ai.microservice.inference

import kotlin.test.Test
import kotlin.test.assertEquals

class ClassificationHeadTest {
    @Test fun `sigmoidTimes100 maps zero logit to 50`() {
        assertEquals(50f, ClassificationHead.sigmoidTimes100(0f), 1e-5f)
    }

    @Test fun `sigmoidTimes100 stays within 0 to 100`() {
        assertEquals(100f, ClassificationHead.sigmoidTimes100(50f), 1e-3f)
        assertEquals(0f, ClassificationHead.sigmoidTimes100(-50f), 1e-3f)
    }
}
