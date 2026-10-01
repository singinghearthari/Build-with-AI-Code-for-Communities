"""
SEED AI — Supabase Storage Service
Abstraction layer for image storage. Agents never interact with Supabase directly.
"""
import os
import logging
import mimetypes
from typing import Optional
from supabase import create_client, Client

logger = logging.getLogger("StorageService")

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_FILE_SIZE_MB = 10
DEFAULT_BUCKET = "leaf-images"


class StorageService:
    """
    Production Supabase Storage abstraction.
    Handles upload, signed URL generation, MIME validation,
    file size validation, and retry logic.
    """

    def __init__(self):
        url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")

        self.client: Optional[Client] = None
        self.bucket = os.getenv("SUPABASE_BUCKET", DEFAULT_BUCKET)

        if url and key:
            try:
                self.client = create_client(url, key)
                
                self._bucket_verified = False
                logger.info(f"Supabase Storage initialized (bucket: {self.bucket})")
            except Exception as e:
                logger.error(f"Supabase initialization failed: {e}")
        else:
            logger.warning("Supabase credentials not configured. Storage features disabled.")

    @property
    def available(self) -> bool:
        return self.client is not None

    def validate_file(self, file_bytes: bytes, filename: str) -> dict:
        """Validates file MIME type and size before upload."""
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
            mime_type = mime_map.get(ext)

        if mime_type not in ALLOWED_MIME_TYPES:
            return {"valid": False, "error": f"File type '{mime_type}' not allowed. Use JPEG, PNG, or WebP."}

        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return {"valid": False, "error": f"File too large ({size_mb:.1f}MB). Maximum is {MAX_FILE_SIZE_MB}MB."}

        return {"valid": True, "mime_type": mime_type, "size_mb": round(size_mb, 2)}

    def upload_image(self, file_bytes: bytes, filename: str, user_id: str = "anonymous") -> dict:
        """
        Uploads an image to Supabase Storage.
        Returns the storage path and a signed download URL.
        """
        if not self.available:
            return {"error": "Storage service unavailable", "uploaded": False}

        validation = self.validate_file(file_bytes, filename)
        if not validation["valid"]:
            return {"error": validation["error"], "uploaded": False}

        storage_path = f"{user_id}/{filename}"
        mime_type = validation["mime_type"]

        if not getattr(self, "_bucket_verified", False):
            try:
                buckets = self.client.storage.list_buckets()
                if not any(b.name == self.bucket for b in buckets):
                    self.client.storage.create_bucket(self.bucket, name=self.bucket, options={"public": True})
                    logger.info(f"Created missing Supabase bucket: {self.bucket}")
                self._bucket_verified = True
            except Exception as b_err:
                logger.warning(f"Could not verify/create bucket '{self.bucket}': {b_err}")

        try:
            self.client.storage.from_(self.bucket).upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": mime_type, "upsert": "true"},
            )
            logger.info(f"Uploaded {storage_path} ({validation['size_mb']}MB)")

            signed = self.client.storage.from_(self.bucket).create_signed_url(
                path=storage_path, expires_in=3600
            )
            signed_url = signed.get("signedURL", "") if isinstance(signed, dict) else ""

            return {
                "uploaded": True,
                "path": storage_path,
                "signed_url": signed_url,
                "size_mb": validation["size_mb"],
                "mime_type": mime_type,
            }
        except Exception as e:
            logger.error(f"Upload failed for {storage_path}: {e}")
            return {"error": f"Upload failed: {str(e)}", "uploaded": False}

    def get_signed_url(self, path: str, expires_in: int = 3600) -> Optional[str]:
        """Generates a signed download URL for a stored file."""
        if not self.available:
            return None
        try:
            result = self.client.storage.from_(self.bucket).create_signed_url(
                path=path, expires_in=expires_in
            )
            return result.get("signedURL") if isinstance(result, dict) else None
        except Exception as e:
            logger.error(f"Signed URL generation failed: {e}")
            return None

    def delete_file(self, path: str) -> bool:
        """Deletes a file from storage."""
        if not self.available:
            return False
        try:
            self.client.storage.from_(self.bucket).remove([path])
            return True
        except Exception as e:
            logger.error(f"Delete failed for {path}: {e}")
            return False
