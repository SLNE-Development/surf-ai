package dev.slne.surf.ai.microservice.inference

import kotlinx.serialization.json.Json
import org.junit.Assume.assumeTrue
import kotlin.io.path.Path
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Cross-consistency check between the Kotlin EmbeddingModel and the Python trainer's Embedder
 * (app/embedding.py). Needs the exported multilingual-e5-small ONNX + tokenizer fixture at
 * src/test/resources/models/embedding/ - not committed (see surf-ai-trainer/README.md for how
 * to generate it locally: `python -c "from pathlib import Path; from app.export_embedding import
 * export; export(Path('../surf-ai-microservice/src/test/resources/models/embedding'))"`).
 * Skipped automatically when the fixture is absent.
 */
class EmbeddingModelFixtureTest {
    @Test fun `embed(hallo welt) matches the recorded Python reference vector`() {
        val onnxPath = Path("src/test/resources/models/embedding/model.onnx")
        val tokenizerPath = Path("src/test/resources/models/embedding/tokenizer.json")
        assumeTrue("embedding model fixture not present locally", onnxPath.exists() && tokenizerPath.exists())

        val referenceVector = Json.decodeFromString<List<Float>>(
            Path("src/test/resources/reference-vectors/hallo-welt.json").readText()
        )

        EmbeddingModel(onnxPath, tokenizerPath, "query: ").use { model ->
            val vector = model.embed(listOf("hallo welt"))[0]
            assertEquals(referenceVector.size, vector.size)
            for (i in vector.indices) {
                assertEquals(referenceVector[i], vector[i], 1e-4f)
            }
        }
    }
}
