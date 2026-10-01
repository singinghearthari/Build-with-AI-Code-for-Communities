"""
SEED AI — Text-to-Speech Service (edge-tts)
Multilingual high-quality voice narration with local file caching.
Supports 10+ Indian languages + English varieties.
Auto-detects language from text input.
"""
import os
import re
import json
import hashlib
import logging
import asyncio
from pathlib import Path
from typing import Optional

logger = logging.getLogger("TTSService")

if os.environ.get("VERCEL"):
    CACHE_DIR = Path("/tmp/tts_cache")
else:
    CACHE_DIR = Path(os.environ.get("TTS_CACHE_DIR", "data/tts_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Unicode ranges for Indian language detection
LANG_RANGES: dict[str, tuple[int, int]] = {
    "hi": (0x0900, 0x097F),   # Hindi / Devanagari
    "mr": (0x0900, 0x097F),   # Marathi (same block)
    "bn": (0x0980, 0x09FF),   # Bengali
    "ta": (0x0B80, 0x0BFF),   # Tamil
    "te": (0x0C00, 0x0C7F),   # Telugu
    "ml": (0x0D00, 0x0D7F),   # Malayalam
    "kn": (0x0C80, 0x0CFF),   # Kannada
    "gu": (0x0A80, 0x0AFF),   # Gujarati
    "pa": (0x0A00, 0x0A7F),   # Gurmukhi (Punjabi)
    "or": (0x0B00, 0x0B7F),   # Odia
    "as": (0x0980, 0x09FF),   # Assamese (same as bn)
    "ur": (0x0600, 0x06FF),   # Urdu / Arabic
}

LANG_TO_SCRIPT: dict[str, str] = {
    "hi": "Devanagari", "mr": "Devanagari", "bn": "Bengali",
    "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "kn": "Kannada",
    "gu": "Gujarati", "pa": "Gurmukhi", "or": "Odia", "as": "Bengali",
    "ur": "Arabic",
}

# Primary female voice per language (preferred, best quality)
LANG_TO_VOICE: dict[str, str] = {
    "en-IN": "en-IN-NeerjaNeural", "en-GB": "en-GB-SoniaNeural", "en-US": "en-US-JennyNeural",
    "hi": "hi-IN-SwaraNeural", "bn": "bn-IN-TanishaaNeural", "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural", "ml": "ml-IN-SobhanaNeural", "kn": "kn-IN-SapnaNeural",
    "gu": "gu-IN-DhwaniNeural", "mr": "mr-IN-AarohiNeural", "ur": "ur-IN-GulNeural",
}

DEFAULT_VOICES = [
    # Indian languages
    {"name": "hi-IN-SwaraNeural", "lang": "hi-IN", "gender": "female", "label": "Swara (Hindi, Female)", "language": "hi"},
    {"name": "hi-IN-MadhurNeural", "lang": "hi-IN", "gender": "male", "label": "Madhur (Hindi, Male)", "language": "hi"},
    {"name": "bn-IN-TanishaaNeural", "lang": "bn-IN", "gender": "female", "label": "Tanishaa (Bengali, Female)", "language": "bn"},
    {"name": "bn-IN-BashkarNeural", "lang": "bn-IN", "gender": "male", "label": "Bashkar (Bengali, Male)", "language": "bn"},
    {"name": "ta-IN-PallaviNeural", "lang": "ta-IN", "gender": "female", "label": "Pallavi (Tamil, Female)", "language": "ta"},
    {"name": "ta-IN-ValluvarNeural", "lang": "ta-IN", "gender": "male", "label": "Valluvar (Tamil, Male)", "language": "ta"},
    {"name": "te-IN-ShrutiNeural", "lang": "te-IN", "gender": "female", "label": "Shruti (Telugu, Female)", "language": "te"},
    {"name": "te-IN-MohanNeural", "lang": "te-IN", "gender": "male", "label": "Mohan (Telugu, Male)", "language": "te"},
    {"name": "ml-IN-SobhanaNeural", "lang": "ml-IN", "gender": "female", "label": "Sobhana (Malayalam, Female)", "language": "ml"},
    {"name": "ml-IN-MidhunNeural", "lang": "ml-IN", "gender": "male", "label": "Midhun (Malayalam, Male)", "language": "ml"},
    {"name": "kn-IN-SapnaNeural", "lang": "kn-IN", "gender": "female", "label": "Sapna (Kannada, Female)", "language": "kn"},
    {"name": "kn-IN-GaganNeural", "lang": "kn-IN", "gender": "male", "label": "Gagan (Kannada, Male)", "language": "kn"},
    {"name": "gu-IN-DhwaniNeural", "lang": "gu-IN", "gender": "female", "label": "Dhwani (Gujarati, Female)", "language": "gu"},
    {"name": "gu-IN-NiranjanNeural", "lang": "gu-IN", "gender": "male", "label": "Niranjan (Gujarati, Male)", "language": "gu"},
    {"name": "mr-IN-AarohiNeural", "lang": "mr-IN", "gender": "female", "label": "Aarohi (Marathi, Female)", "language": "mr"},
    {"name": "mr-IN-ManoharNeural", "lang": "mr-IN", "gender": "male", "label": "Manohar (Marathi, Male)", "language": "mr"},
    {"name": "ur-IN-GulNeural", "lang": "ur-IN", "gender": "female", "label": "Gul (Urdu, Female)", "language": "ur"},
    {"name": "ur-IN-SalmanNeural", "lang": "ur-IN", "gender": "male", "label": "Salman (Urdu, Male)", "language": "ur"},
    # English varieties
    {"name": "en-IN-NeerjaNeural", "lang": "en-IN", "gender": "female", "label": "Neerja (Indian English, Female)", "language": "en"},
    {"name": "en-IN-PrabhatNeural", "lang": "en-IN", "gender": "male", "label": "Prabhat (Indian English, Male)", "language": "en"},
    {"name": "en-GB-SoniaNeural", "lang": "en-GB", "gender": "female", "label": "Sonia (British English, Female)", "language": "en"},
    {"name": "en-GB-RyanNeural", "lang": "en-GB", "gender": "male", "label": "Ryan (British English, Male)", "language": "en"},
    {"name": "en-US-JennyNeural", "lang": "en-US", "gender": "female", "label": "Jenny (American English, Female)", "language": "en"},
    {"name": "en-US-GuyNeural", "lang": "en-US", "gender": "male", "label": "Guy (American English, Male)", "language": "en"},
]

LANGUAGES: list[dict] = [
    {"code": "en", "label": "English", "native": "English", "region": "Global"},
    {"code": "hi", "label": "Hindi", "native": "हिन्दी", "region": "India"},
    {"code": "bn", "label": "Bengali", "native": "বাংলা", "region": "India"},
    {"code": "ta", "label": "Tamil", "native": "தமிழ்", "region": "India"},
    {"code": "te", "label": "Telugu", "native": "తెలుగు", "region": "India"},
    {"code": "ml", "label": "Malayalam", "native": "മലയാളം", "region": "India"},
    {"code": "kn", "label": "Kannada", "native": "ಕನ್ನಡ", "region": "India"},
    {"code": "gu", "label": "Gujarati", "native": "ગુજરાતી", "region": "India"},
    {"code": "mr", "label": "Marathi", "native": "मराठी", "region": "India"},
    {"code": "ur", "label": "Urdu", "native": "اردو", "region": "India"},
]


def detect_language(text: str) -> str:
    """Detect the primary language of the given text based on Unicode script ranges."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in LANG_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break

    if not counts:
        return "en"

    # Disambiguate Devanagari (hi vs mr)
    if "hi" in counts and "mr" in counts:
        del counts["mr"]

    # Disambiguate Bengali block (bn vs as)
    if "bn" in counts and "as" in counts:
        del counts["as"]

    return max(counts, key=counts.get)


class TTSService:
    _instance = None

    def __init__(self):
        self._available = False
        self._voices: list[dict] = []
        self._edge_available = False
        self._lang_voice_override: dict[str, str] = {}
        self._init()

    def _init(self):
        try:
            import edge_tts
            self._edge_available = True
            self._available = True
            self._voices = self._discover_voices()
            logger.info(f"edge-tts available — {len(self._voices)} multilingual voices loaded")
        except ImportError:
            logger.warning("edge-tts not installed. Run: pip install edge-tts")
            self._voices = list(DEFAULT_VOICES)
            self._available = False

    def _discover_voices(self) -> list[dict]:
        try:
            import asyncio
            import edge_tts
            loop = asyncio.new_event_loop()
            raw = loop.run_until_complete(edge_tts.list_voices())
            loop.close()
            discovered = []
            for v in raw:
                locale = v["Locale"]
                name = v["ShortName"]
                gender = v.get("Gender", "Female").lower()
                discovered.append({
                    "name": name,
                    "lang": locale,
                    "gender": gender,
                    "label": f"{v['FriendlyName']} ({locale})",
                    "language": locale.split("-")[0].lower(),
                })
            if discovered:
                return discovered
        except Exception as e:
            logger.warning(f"Voice discovery failed: {e}")
        return list(DEFAULT_VOICES)

    @classmethod
    def get_instance(cls) -> "TTSService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_available(self) -> bool:
        return self._available

    def get_voices(self, language: Optional[str] = None) -> list[dict]:
        if language:
            lang_lower = language.lower().split("-")[0]
            return [v for v in self._voices if v["language"] == lang_lower]
        return list(self._voices)

    def get_languages(self) -> list[dict]:
        return list(LANGUAGES)

    def get_voice_for_language(self, lang_code: str) -> str:
        lang_key = lang_code.lower().split("-")[0]
        override = self._lang_voice_override.get(lang_code)
        if override:
            return override
        if lang_key in LANG_TO_VOICE:
            return LANG_TO_VOICE[lang_key]
        return "en-IN-NeerjaNeural"

    def detect_language(self, text: str) -> str:
        return detect_language(text)

    def _cache_key(self, text: str, voice: str, rate: str) -> str:
        raw = f"{text}||{voice}||{rate}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.mp3"

    def _cache_meta_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.json"

    async def generate_async(self, text: str, voice: Optional[str] = None, rate: str = "+0%") -> dict:
        if not self._available:
            return {"status": "error", "message": "edge-tts not installed"}

        detected_lang = self.detect_language(text)
        resolved_voice = voice or self.get_voice_for_language(detected_lang)

        key = self._cache_key(text, resolved_voice, rate)
        cached = self._cache_path(key)

        if cached.exists():
            meta = {"cached": True, "voice": resolved_voice, "rate": rate,
                    "detected_language": detected_lang, "path": str(cached), "size": cached.stat().st_size}
            logger.info(f"TTS cache hit: {key[:12]}... ({cached.stat().st_size} bytes, {resolved_voice})")
            return {"status": "success", **meta}

        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, resolved_voice, rate=str(rate))
            temp_path = cached.with_suffix(".tmp")
            await communicate.save(str(temp_path))
            temp_path.rename(cached)

            meta = {"cached": False, "voice": resolved_voice, "rate": rate,
                    "detected_language": detected_lang, "path": str(cached), "size": cached.stat().st_size}
            self._cache_meta_path(key).write_text(json.dumps(
                {"text_len": len(text), "voice": resolved_voice, "rate": rate, "detected_language": detected_lang}
            ))
            logger.info(f"TTS generated: {key[:12]}... ({cached.stat().st_size} bytes, {resolved_voice})")
            return {"status": "success", **meta}
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return {"status": "error", "message": str(e)}

    def generate(self, text: str, voice: Optional[str] = None, rate: str = "+0%") -> dict:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.generate_async(text, voice, rate))
        finally:
            loop.close()

    def clear_cache(self, max_age_days: int = 7) -> int:
        import time
        now = time.time()
        cutoff = now - (max_age_days * 86400)
        removed = 0
        for f in CACHE_DIR.iterdir():
            if f.suffix in (".mp3", ".json", ".tmp") and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        if removed:
            logger.info(f"TTS cache cleaned: {removed} files removed")
        return removed
