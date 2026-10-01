"""
SEED AI — Waste to Wealth Agent (Production)
Identifies revenue opportunities from agricultural waste streams.
Returns structured AgentResult — never fakes success.
"""
from typing import Dict, Any
from pydantic import BaseModel
from .base_agent import BaseAgent
from utils.dataset_manager import DatasetManager


class WasteOpportunity(BaseModel):
    waste_type: str
    conversion_method: str
    output_product: str
    estimated_revenue_per_ton: str
    required_investment: str
    difficulty_level: str


class WasteToWealthResult(BaseModel):
    waste_streams: list[str]
    opportunities: list[WasteOpportunity]
    total_potential_revenue: str
    equipment_needed: list[str]
    government_subsidies: list[str]
    environmental_benefits: list[str]
    quick_wins: list[str]
    reasoning: str


class WasteToWealthAgent(BaseAgent):

    def __init__(self):
        super().__init__("WasteToWealth")
        self.dataset_manager = DatasetManager()

    def _process(self, context: Dict[str, Any]) -> tuple:
        location = context.get("location", "")
        crop = context.get("crop", "")
        budget = context.get("budget", 0)
        query = context.get("text_query", "")
        self.log_execution(f"Analyzing waste-to-wealth for crop={crop}, location={location}")

        tool_calls = ["Knowledge Base (crops)", "Knowledge Base (government_schemes)"]
        waste_context = ""

        crop_data = self.dataset_manager.query("crops", crop)
        scheme_data = self.dataset_manager.query("government_schemes", "waste")

        prompt = f"""
You are an expert in agricultural waste management and circular economy, specializing in Indian farming.

Farmer's Query: {query}
Location: {location}
Crop: {crop}
Available Budget for Investment: ₹{budget}
{waste_context}
Crop Knowledge Base Data: {crop_data[:3] if crop_data else "No specific data available"}
Government Schemes Data: {scheme_data[:3] if scheme_data else "No specific data available"}

Analyze the agricultural waste streams from this crop and identify wealth-generation opportunities:

1. Available waste streams from {crop} (e.g., stalks, husks, leaves, roots, etc.)
2. For each viable opportunity, provide:
   - Waste type being converted
   - Conversion method (composting, biochar, briquettes, mushroom cultivation, animal feed, vermicompost, biofuel, paper/packaging, etc.)
   - Output product
   - Estimated revenue per ton of waste processed (in ₹)
   - Required investment to start (in ₹)
   - Difficulty level (Easy/Medium/Hard)
3. Total potential annual revenue from waste conversion
4. Equipment needed (list specific items)
5. Available government subsidies and schemes for waste processing
6. Environmental benefits (carbon credits, soil health, water conservation)
7. Quick wins — things the farmer can start doing THIS WEEK with minimal investment

Focus on practical, actionable opportunities. Prioritize low-investment, high-return options suitable for small/medium Indian farmers.
"""
        response = self.call_llm(prompt, schema=WasteToWealthResult)
        result = WasteToWealthResult.model_validate_json(response.text)
        tokens = response.total_tokens

        has_kb_data = bool(crop_data or scheme_data)
        return (
            result.model_dump(),
            tool_calls,
            tokens,
            78.0 if has_kb_data else 62.0,
            result.reasoning,
        )
