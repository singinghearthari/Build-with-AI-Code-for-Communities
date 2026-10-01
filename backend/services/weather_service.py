import os
import time
import logging
from typing import Dict, Tuple
from utils.api_gateway import APIGateway

logger = logging.getLogger("WeatherService")

_WEATHER_CACHE: Dict[str, Tuple[dict, float]] = {}
WEATHER_CACHE_TTL = 600  # 10 minutes cache


class WeatherService:
    """
    Production weather service using OpenWeatherMap API with in-memory caching.
    Returns live weather data or a structured error if unavailable.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, location: str) -> dict:
        if not self.api_key:
            logger.warning("OpenWeatherMap API Key not configured.")
            return {"error": "OPENWEATHER_API_KEY not set", "source": "none"}

        loc_key = (location or "").strip().lower()
        now = time.time()
        if loc_key in _WEATHER_CACHE:
            cached_data, cached_time = _WEATHER_CACHE[loc_key]
            if now - cached_time < WEATHER_CACHE_TTL:
                logger.info(f"Weather cache hit for '{location}'")
                return cached_data

        try:
            params = {"q": location, "appid": self.api_key, "units": "metric"}
            data = APIGateway.fetch_json(self.base_url, params=params, timeout=5)

            result = {
                "rain_1h": data.get("rain", {}).get("1h", 0),
                "humidity": data["main"]["humidity"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "wind_speed": data["wind"]["speed"],
                "description": data["weather"][0]["description"] if data.get("weather") else "Unknown",
                "source": "openweathermap",
            }
            _WEATHER_CACHE[loc_key] = (result, now)
            return result
        except Exception as e:
            logger.error(f"Weather API call failed: {e}")
            return {"error": f"Weather service unavailable: {str(e)}", "source": "none"}
