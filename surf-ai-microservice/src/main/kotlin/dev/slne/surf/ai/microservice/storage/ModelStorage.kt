package dev.slne.surf.ai.microservice.storage

import dev.slne.surf.ai.microservice.config.AiConfig
import io.minio.GetObjectArgs
import io.minio.MinioClient
import io.minio.StatObjectArgs
import io.minio.errors.ErrorResponseException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import kotlin.io.path.createParentDirectories

class ModelStorage(private val endpoint: String, private val bucket: String, accessKey: String, secretKey: String) {
    constructor(config: AiConfig) : this(config.s3Endpoint, config.s3Bucket, config.s3AccessKey, config.s3SecretKey)

    private val client = MinioClient.builder().endpoint(endpoint).credentials(accessKey, secretKey).build()

    suspend fun download(key: String, dest: Path) = withContext(Dispatchers.IO) {
        dest.createParentDirectories()
        client.getObject(GetObjectArgs.builder().bucket(bucket).`object`(key).build()).use { input ->
            Files.copy(input, dest, StandardCopyOption.REPLACE_EXISTING)
        }
        Unit
    }

    suspend fun exists(key: String): Boolean = withContext(Dispatchers.IO) {
        try {
            client.statObject(StatObjectArgs.builder().bucket(bucket).`object`(key).build())
            true
        } catch (e: ErrorResponseException) {
            if (e.errorResponse().code() == "NoSuchKey") false else throw e
        }
    }
}
