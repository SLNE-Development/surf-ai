package dev.slne.surf.ai.microservice.storage

import io.minio.BucketExistsArgs
import io.minio.MakeBucketArgs
import io.minio.MinioClient
import io.minio.PutObjectArgs
import kotlinx.coroutines.runBlocking
import java.io.ByteArrayInputStream
import kotlin.io.path.createTempFile
import kotlin.io.path.readBytes
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ModelStorageTest {
    private val endpoint = "http://localhost:9000"
    private val bucket = "surf-ai"
    private val accessKey = "surfai"
    private val secretKey = "surfaikey"

    private val rawClient = MinioClient.builder().endpoint(endpoint).credentials(accessKey, secretKey).build()

    @Test fun `download fetches bytes previously put via the raw client`() = runBlocking {
        if (!rawClient.bucketExists(BucketExistsArgs.builder().bucket(bucket).build())) {
            rawClient.makeBucket(MakeBucketArgs.builder().bucket(bucket).build())
        }

        val key = "models/test/probe-${System.nanoTime()}.bin"
        val bytes = "hello model storage".toByteArray()
        rawClient.putObject(
            PutObjectArgs.builder()
                .bucket(bucket)
                .`object`(key)
                .stream(ByteArrayInputStream(bytes), bytes.size.toLong(), -1)
                .build()
        )

        val storage = ModelStorage(endpoint, bucket, accessKey, secretKey)
        assertTrue(storage.exists(key))
        assertFalse(storage.exists("models/test/does-not-exist.bin"))

        val dest = createTempFile()
        storage.download(key, dest)
        assertEquals(bytes.toList(), dest.readBytes().toList())
    }
}
