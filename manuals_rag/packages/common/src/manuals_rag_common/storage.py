from __future__ import annotations

from datetime import timedelta
from io import BytesIO

from minio.error import S3Error
from minio import Minio
from minio.commonconfig import CopySource

from manuals_rag_common.config import settings


class ObjectStore:
    def __init__(self) -> None:
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            region=settings.minio_region,
        )
        self.public_client = Minio(
            settings.minio_public_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_public_secure,
            region=settings.minio_region,
        )

    def ensure_buckets(self) -> None:
        for bucket in (settings.minio_bucket_originals, settings.minio_bucket_artifacts):
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)

    def put_bytes(self, bucket: str, object_name: str, data: bytes, content_type: str) -> str:
        self.client.put_object(
            bucket,
            object_name,
            BytesIO(data),
            len(data),
            content_type=content_type,
        )
        return f"s3://{bucket}/{object_name}"

    def object_exists(self, bucket: str, object_name: str) -> bool:
        try:
            self.client.stat_object(bucket, object_name)
            return True
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
                return False
            raise

    def remove_object(self, bucket: str, object_name: str) -> None:
        self.client.remove_object(bucket, object_name)

    def copy_object(self, bucket: str, source_object_name: str, target_object_name: str) -> str:
        self.client.copy_object(
            bucket,
            target_object_name,
            CopySource(bucket, source_object_name),
        )
        return f"s3://{bucket}/{target_object_name}"

    def presigned_get_url(self, bucket: str, object_name: str, *, expires: timedelta = timedelta(hours=1)) -> str:
        return self.public_client.presigned_get_object(bucket, object_name, expires=expires)
