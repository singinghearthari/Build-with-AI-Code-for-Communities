"""
SEED AI — Hugging Face Serverless Vision Analyzer
Drop-in upgrade from Gemini Vision for plant disease diagnostics and waste classification.
Uses huggingface_hub InferenceClient for zero-infrastructure serverless inference.
Includes proactive connectivity check to surface DNS/network errors early.
"""
import os
import time
import logging
import socket
from io import BytesIO
from typing import Optional
from huggingface_hub import InferenceClient
from PIL import Image
import requests

logger = logging.getLogger("HFVisionAnalyzer")

HF_API_HOST = "api-inference.huggingface.co"
HF_API_BASE = f"https://{HF_API_HOST}"


def _check_connectivity() -> tuple[bool, str]:
    """Deprecated: HF Inference API is disabled."""
    return False, "Hugging Face Inference API is disabled."


class HFVisionAnalyzer:
    _instance = None

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = ""
        self.client = None
        self._connectivity_error = "HF Vision Analyzer is disabled."

        self.models = {
            "plant_disease_vit": "wambugu71/crop_leaf_diseases_vit",
            "plant_disease_mobilenet": "Daksh159/plant-disease-mobilenetv2",
            "waste_siglip": "prithivMLmods/Augmented-Waste-Classifier-SigLIP2",
        }

    @classmethod
    def get_instance(cls) -> "HFVisionAnalyzer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_available(self) -> bool:
        return self.client is not None

    def connectivity_error(self) -> str | None:
        return getattr(self, "_connectivity_error", None)

    def _prepare_image(self, image_source) -> bytes:
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                resp = requests.get(image_source, timeout=30)
                resp.raise_for_status()
                return resp.content
            with open(image_source, "rb") as f:
                return f.read()
        elif isinstance(image_source, Image.Image):
            buf = BytesIO()
            image_source.save(buf, format="JPEG")
            return buf.getvalue()
        elif isinstance(image_source, bytes):
            return image_source
        raise TypeError("Unsupported image source. Use path, URL, bytes, or PIL Image.")

    def analyze_crop_health(self, image_source, use_vit: bool = True, retries: int = 2) -> dict:
        model = self.models["plant_disease_vit"] if use_vit else self.models["plant_disease_mobilenet"]
        if not self.client:
            reason = self._connectivity_error or "HF client not initialized — check HUGGING_FACE_API"
            return {"status": "error", "message": reason}

        try:
            image_bytes = self._prepare_image(image_source)
        except Exception as e:
            return {"status": "error", "message": f"Image preparation failed: {e}"}

        last_error = None
        for attempt in range(retries + 1):
            try:
                start = time.time()
                predictions = self.client.image_classification(image_bytes, model=model)
                latency_ms = (time.time() - start) * 1000
                return {
                    "status": "success",
                    "predictions": predictions,
                    "model": model,
                    "latency_ms": round(latency_ms, 1),
                }
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "503" in error_str and attempt < retries:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"HF model cold-start (503), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                break

        error_msg = str(last_error) or type(last_error).__name__
        if not str(last_error).strip():
            error_msg = f"{type(last_error).__name__} — check network connectivity to {HF_API_HOST}"
        return {"status": "error", "message": error_msg}

    def classify_waste(self, image_source, retries: int = 2) -> dict:
        model = self.models["waste_siglip"]
        if not self.client:
            reason = self._connectivity_error or "HF client not initialized — check HUGGING_FACE_API"
            return {"status": "error", "message": reason}

        try:
            image_bytes = self._prepare_image(image_source)
        except Exception as e:
            return {"status": "error", "message": f"Image preparation failed: {e}"}

        last_error = None
        for attempt in range(retries + 1):
            try:
                start = time.time()
                predictions = self.client.image_classification(image_bytes, model=model)
                latency_ms = (time.time() - start) * 1000
                return {
                    "status": "success",
                    "predictions": predictions,
                    "model": model,
                    "latency_ms": round(latency_ms, 1),
                }
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "503" in error_str and attempt < retries:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"HF model cold-start (503), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                break

        error_msg = str(last_error) or type(last_error).__name__
        if not str(last_error).strip():
            error_msg = f"{type(last_error).__name__} — check network connectivity to {HF_API_HOST}"
        return {"status": "error", "message": error_msg}
