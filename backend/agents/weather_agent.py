"""
SEED AI — Weather Agent (Production)
Fetches live weather data, then uses Gemini to assess agricultural risk.
Returns structured AgentResult — never fakes success.
"""
import asyncio
from typing import Dict, Any
from pydantic import BaseModel
from .base_agent import BaseAgent
from services.weather_service import WeatherService


class WeatherRiskAssessment(BaseModel):
    rain_probability: float
    humidity: float
    temperature: float
    wind_speed: float
    weather_risk: str
    spray_safe: bool
    irrigation_needed: bool
    reasoning: str


class WeatherAgent(BaseAgent):

    def __init__(self):
        super().__init__("Weather")
        self.weather_service = WeatherService()

    async def _process_async(self, context: Dict[str, Any]) -> tuple:
        location = context.get("location", "Bangalore")
        crop = context.get("crop", "general crops")
        self.log_execution(f"Fetching weather data for {location}")

        raw_weather = await asyncio.to_thread(self.weather_service.get_weather, location)
        tool_calls = ["OpenWeatherMap API"]

        if "error" in raw_weather:
            self.log_execution("Weather API failed, using LLM general knowledge")
            raw_weather = {"source": "gemini_fallback", "note": "Live data unavailable"}
            tool_calls.append("LLM Fallback")

        prompt = f"""
You are an agricultural weather risk analyst.

Location: {location}
Crop: {crop}
Weather data: {raw_weather}

Based on this weather data, provide a complete agricultural weather risk assessment.
Determine: rain probability (0-100), humidity, temperature, wind speed,
overall weather risk description, whether it is safe to spray pesticides,
whether irrigation is needed, and your reasoning.

If weather data is unavailable, use your knowledge of typical weather for this location and season.
"""
        response = await self.async_call_llm(prompt, schema=WeatherRiskAssessment)
        result = WeatherRiskAssessment.model_validate_json(response.text)
        tokens = response.total_tokens

        return (
            result.model_dump(),
            tool_calls,
            tokens,
            85.0 if raw_weather.get("source") != "gemini_fallback" else 50.0,
            result.reasoning,
        )

    def _process(self, context: Dict[str, Any]) -> tuple:
        location = context.get("location", "Bangalore")
        crop = context.get("crop", "general crops")
        self.log_execution(f"Fetching weather data for {location}")

        # Step 1: Get live weather data
        raw_weather = self.weather_service.get_weather(location)
        tool_calls = ["OpenWeatherMap API"]

        if "error" in raw_weather:
            self.log_execution("Weather API failed, using Gemini general knowledge")
            raw_weather = {"source": "gemini_fallback", "note": "Live data unavailable"}
            tool_calls.append("Gemini Fallback")

        # Step 2: Use Gemini to assess agricultural risk
        prompt = f"""
You are an agricultural weather risk analyst.

Location: {location}
Crop: {crop}
Weather data: {raw_weather}

Based on this weather data, provide a complete agricultural weather risk assessment.
Determine: rain probability (0-100), humidity, temperature, wind speed,
overall weather risk description, whether it is safe to spray pesticides,
whether irrigation is needed, and your reasoning.

If weather data is unavailable, use your knowledge of typical weather for this location and season.
"""
        response = self.call_llm(prompt, schema=WeatherRiskAssessment)
        result = WeatherRiskAssessment.model_validate_json(response.text)
        tokens = response.total_tokens

        return (
            result.model_dump(),
            tool_calls,
            tokens,
            85.0 if raw_weather.get("source") != "gemini_fallback" else 50.0,
            result.reasoning,
        )
