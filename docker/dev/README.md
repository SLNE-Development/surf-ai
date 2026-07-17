# Local dev infrastructure

Start Postgres, RabbitMQ, and MinIO for local development:

```bash
docker compose -f docker/dev/docker-compose.yml up -d
```

## Credentials / ports

| Service    | Host                    | User/Access Key | Password/Secret Key |
|------------|-------------------------|------------------|----------------------|
| Postgres   | `localhost:5432`        | `surf_ai`        | `surf_ai`            |
| RabbitMQ   | `localhost:5672`        | `surf`           | `surf`               |
| RabbitMQ UI| `localhost:15672`       | `surf`           | `surf`               |
| MinIO API  | `localhost:9000`        | `surfai`         | `surfaikey`          |
| MinIO UI   | `localhost:9091`        | `surfai`         | `surfaikey`          |

Database name: `surf_ai`. MinIO bucket `surf-ai` is auto-created by the `minio-init` container.
