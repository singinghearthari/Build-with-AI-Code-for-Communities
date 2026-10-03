<div align="center">

<br/>

<!-- Hero Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=SEED%20AI&fontSize=80&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Smart%20Ecosystem%20for%20Enhanced%20Agricultural%20Decisions&descAlignY=60&descSize=18" width="100%"/>

<br/>

<!-- Badges Row 1 -->
<p>
  <img src="https://img.shields.io/badge/Google%20AI%20Agents-Capstone%202026-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/Multi--Agent-Architecture-FF6B35?style=for-the-badge&logo=robot&logoColor=white"/>
  <img src="https://img.shields.io/badge/15%20AI%20Agents-Autonomous-22C55E?style=for-the-badge&logo=sparkles&logoColor=white"/>
</p>

<!-- Badges Row 2 -->
<p>
  <img src="https://img.shields.io/badge/Next.js-16.2.9-000000?style=for-the-badge&logo=nextdotjs&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TypeScript-5.x-3178C6?style=for-the-badge&logo=typescript&logoColor=white"/>
</p>

<!-- Badges Row 3 -->
<p>
  <img src="https://img.shields.io/badge/Gemini-2.5%20Flash-8B5CF6?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLO11-Vision%20Pipeline-FF0000?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Firebase-Auth%20%2B%20Firestore-FFCA28?style=for-the-badge&logo=firebase&logoColor=black"/>
  <img src="https://img.shields.io/badge/MCP-Model%20Context%20Protocol-6366F1?style=for-the-badge&logo=protocol&logoColor=white"/>
</p>

<br/>

> **Autonomous Agricultural Intelligence — 15 AI agents that observe, reason, plan, and act so farmers don't have to choose between information overload and flying blind.**

<br/>

---

</div>

## 📖 Table of Contents

- [🌾 Overview](#-overview)
- [🚨 The Problem](#-the-problem)
- [💡 The Solution](#-the-solution)
- [🏗️ System Architecture](#️-system-architecture)
- [🤖 The Agent Ecosystem](#-the-agent-ecosystem)
- [⚙️ Execution Modes](#️-execution-modes)
- [🔄 Multi-Agent Workflow](#-multi-agent-workflow)
- [✨ Core Capabilities](#-core-capabilities)
- [🛠️ Technology Stack](#️-technology-stack)
- [🔌 MCP Server](#-mcp-server)
- [📡 Real-Time Streaming](#-real-time-streaming)
- [🔒 Security & Isolation](#-security--isolation)
- [⚡ Performance](#-performance)
- [🎯 Example Scenario](#-example-scenario)
- [🖥️ User Interface](#️-user-interface)
- [🛡️ Responsible AI](#️-responsible-ai)
- [📊 Impact Metrics](#-impact-metrics)

---

## 🌾 Overview

**SEED AI** is an *Autonomous Agricultural Intelligence System* built on a **Multi-Agent AI Architecture**. It goes far beyond chatbots — it orchestrates 15 specialized AI agents that collaborate in real time to analyze crop conditions, retrieve live environmental data, generate personalized action plans, and continuously adapt based on farm history.

```
Farmers don't need more information.
They need timely, personalized decisions.
```

Built as the **Google AI Agents: Intensive Vibe Coding Capstone — July 2026**, SEED AI demonstrates how autonomous agents applying planning, reasoning, tool use, memory, reflection, and guardrails can transform agriculture from reactive guesswork into proactive, data-driven farm management.

---

## 🚨 The Problem

Farmers today face a fragmented digital landscape — weather apps, disease forums, market portals, and government websites that never talk to each other.

| Challenge | Reality |
|-----------|---------|
| 🔬 **Delayed disease diagnosis** | Visual symptoms are frequently misidentified without expert access |
| 🌧️ **Unpredictable weather impact** | Rain and humidity affect treatment efficacy in real time |
| 📋 **No personalized guidance** | Generic advice ignores specific crops, soils, locations, and budgets |
| 🧩 **Fragmented tools** | Weather, prices, schemes, diagnostics — all separate, all manual to combine |
| 💸 **Budget-treatment tradeoffs** | No tool compares effectiveness vs. cost transparently |
| 🧠 **No memory across sessions** | Every query starts from zero — farm history is lost |
| 📜 **Missed government subsidies** | Eligible farmers miss PMFBY, PM-KISAN, KCC due to poor awareness |

---

## 💡 The Solution

SEED AI introduces an **autonomous multi-agent ecosystem** where specialized agents collaborate to generate personalized farming decisions:

<div align="center">

| Capability | What SEED AI Does |
|------------|-------------------|
| 👁️ **Observes** | Accepts text, voice (11 languages, auto-detect), and images (drag-drop or camera) |
| 🧠 **Reasons** | Uses Gemini Function Calling to dynamically route queries to the right agents |
| 📅 **Plans** | Generates 7-day action schedules with Critical / High / Medium / Low priorities |
| 🔧 **Calls Tools** | Integrates 8 external services — weather, markets, vision models, database |
| 🤝 **Coordinates Agents** | Up to 13 specialized agents + memory + translation, running concurrently |
| 📋 **Generates Reports** | Fuses all agent outputs into one structured plan with confidence scores |
| 🔁 **Learns** | Persistent Firestore memory across sessions — farm profile, disease history, traces |
| 🛡️ **Applies Guardrails** | Rain-check delays sprays; budget ceiling warns and suggests alternatives |
| 📡 **Streams Live** | Server-Sent Events show every agent step, latency, and status in real time |

</div>

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                            FARMER                                │
│               ┌───────────┼───────────┐                          │
│               ▼           ▼           ▼                          │
│             Voice       Image       Text                         │
│        (Web Speech)   (Upload)    (Query)                        │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│               FRONTEND  (Next.js 16 + Tailwind 4)                │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ ┌────────┐   │
│  │Simulator │ │Dashboard │ │Activity│ │Analytics │ │History │   │
│  └──────────┘ └──────────┘ └────────┘ └──────────┘ └────────┘   │
│  SSE Client · SpeechRecognition (11 lang) · TTS · Geolocation    │
│  shadcn/ui · Framer Motion · Recharts · Lucide React             │
└───────────────────────────┬──────────────────────────────────────┘
                            │  SSE / REST  (Firebase Auth Bearer)
┌───────────────────────────▼──────────────────────────────────────┐
│                BACKEND  (FastAPI + Python 3.10+)                  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │            GEMINI ORCHESTRATOR  (1195 lines)               │  │
│  │  Routing Engine · Tier 1 Execution · Fusion Engine         │  │
│  │  Guardrails · Reflection · EventBus · Evaluator            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Vision   │ │ Weather  │ │   Crop   │ │  Soil &  │  ...+9     │
│  │ Hybrid   │ │ OWM +    │ │Knowledge │ │ Nutrient │  more      │
│  │ Pipeline │ │ Gemini   │ │ RAG + KB │ │ 5 soils  │  agents    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
│                                                                   │
│  YOLO11 · OpenRouter Gemma-4 26B · Qwen2.5-VL · HuggingFace     │
│  edge-tts · Supabase Storage · Firebase Admin · MCP Server       │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                     EXTERNAL SERVICES                             │
│   Gemini (11 keys)  ·  HuggingFace  ·  OpenRouter  ·  OWM       │
│   Data.gov.in Agmarknet  ·  Firebase  ·  Supabase  ·  g4f       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🤖 The Agent Ecosystem

<div align="center">

| # | Agent | Role | Key Output |
|---|-------|------|------------|
| 1 | 🔬 **Vision** | Hybrid disease diagnosis (YOLO11 + LLMs) | Disease name, severity %, confidence |
| 2 | 🌤️ **Weather** | Live risk assessment via OpenWeatherMap | Spray-safe bool, rain probability |
| 3 | 🌿 **Crop Knowledge** | RAG over disease/treatment knowledge base | Treatment protocols, dosages |
| 4 | 🌱 **Soil & Nutrient** | Soil health analysis (5 soil types) | pH, N/P/K, health score 0–100 |
| 5 | 💰 **Budget** | Cost-optimized treatment selection | Cheapest & best-value options |
| 6 | 📈 **Market** | Live Agmarknet prices + MSP data | Price trend, sell recommendation |
| 7 | 🏛️ **Government Scheme** | 10 eligible schemes matched | PMFBY, PM-KISAN, KCC details |
| 8 | 📅 **Task Planning** | 7-day personalized schedule | Daily tasks with priorities |
| 9 | 💧 **Irrigation** | 7-day water schedule | Method, litres/acre, timing |
| 10 | 🐛 **Entomologist** | 6-pest management strategies | Cultural + biological + chemical |
| 11 | ♻️ **Waste-to-Wealth** | Agricultural waste revenue | Revenue opportunities, quick wins |
| 12 | 🌾 **Crop Prediction** | Optimal crop selection | Top recommendations + risk factors |
| 13 | 🔮 **Disease Prediction** | Preventive risk scoring | Predicted diseases before symptoms |
| 14 | 🧠 **Memory** | Firestore persistent farm memory | Profile, history, traces |
| 15 | 🌐 **Translation** | Tamil ↔ English bilingual | 11-language speech, 12-language TTS |

</div>

---

## ⚙️ Execution Modes

| Mode | Agents | Best For |
|------|--------|----------|
| 🤖 **Auto** | Tier 1 core + LLM-chosen specialists | General queries |
| ⚡ **Fast Track** | Vision + Disease Prediction only | Urgent leaf diagnosis |
| 🎯 **Ensemble** | 5-agent coordinated swarm | Comprehensive health check |
| 🔍 **Specific** | Single user-selected agent | Targeted expert question |
| 🌐 **Swarm (All)** | All 13 agents end-to-end | Complete farm assessment |

---

## 🔄 Multi-Agent Workflow

```
Step 1 ── User Input
         Text · Voice (→ Speech → Text) · Leaf Image
         Location / Crop / Budget auto-inferred from context

         ▼
Step 2 ── Orchestrator
         Cache check (md5, 300s TTL)
         Memory load from Firestore (farm profile, history)
         Context inference (text patterns → memory → geolocation)
         Gemini Function Calling → dynamic agent routing
                  ▼
         ┌─ Tier 1: Concurrent (asyncio.gather, 60s timeout) ─────┐
         │  crop_knowledge  ·  weather  ·  soil_nutrient          │
         │  budget  ·  market  ·  [LLM-chosen specialists]        │
         └────────────────────────────────────────────────────────┘
         ▼
Step 3 ── Fusion & Reflection
         Decision Fusion via Gemini LLMFusionSchema
         └→ Fallback: local _fusion() if LLM fails
         Guardrails:
           ├─ 🌧  Rain > 60% → postpone chemical sprays
           └─ 💸  Cost > 110% budget → warn + suggest alternatives
         Reflection: budget compliance · spray safety · contradictions
                  ▼
         ┌─ Tier 2: Fire-and-forget (background) ────────────────┐
         │  government_schemes · waste_to_wealth · task_planning  │
         └────────────────────────────────────────────────────────┘
         ▼
Step 4 ── Output
         EventBus → Batch Firestore persist (2s flush / 50 events)
         AgentEvaluator → per-user traces + JSON logs
         SSE Stream → frontend (events → pre_result → result)
         6 Result Tabs: Action Plan · Timeline · Weather · Soil · Markets · Waste
```

---

## ✨ Core Capabilities

<details>
<summary><b>🔬 Crop Disease Diagnosis — Hybrid Vision Pipeline</b></summary>

A 5-model cascade with automatic failover ensures the best possible diagnosis:

| Model | Provider | Role |
|-------|----------|------|
| **YOLO11n** | Local (CUDA/MPS/CPU) | 17 disease + 10 waste classes, bounding boxes |
| **Gemma-4 26B** | OpenRouter (free) | Primary structured JSON disease analysis |
| **Gemini Vision 2.0 Flash** | Google | Vision fallback |
| **Qwen2.5-VL-7B** | HuggingFace Inference | Expert analysis with latency tracking |
| **ViT + MobileNetV2 + SigLIP2** | HuggingFace serverless | Classification + waste detection |

**Output:** Disease name · Severity (Mild/Moderate/Severe) · Confidence % · Affected area % · Immediate action

</details>

<details>
<summary><b>🌤️ Weather Intelligence</b></summary>

- Live data via OpenWeatherMap — rain probability, humidity, temperature, wind speed
- Agricultural risk assessed via Gemini structured schema
- `spray_safe` boolean computed automatically
- Live confidence: **85%** · Fallback (general knowledge): **50%**

</details>

<details>
<summary><b>💰 Budget-Aware Treatment Planning</b></summary>

- 40+ treatments across 12 diseases, 8 fertilizers with INR prices in knowledge base
- Cheapest vs. best-value side-by-side comparison
- Automatic budget ceiling warning (>110%) with low-cost alternatives
- Example: Mancozeb 75% WP at ₹280 vs. Azoxystrobin at ₹800

</details>

<details>
<summary><b>📈 Market Intelligence</b></summary>

- Live prices from Data.gov.in Agmarknet (20,000+ records)
- MSP data for 26 crops (17 Kharif, 6 Rabi, 3 Other) with cost of production
- 8 major APMC mandis tracked: Azadpur Delhi, Vashi Mumbai, Lasalgaon Nashik, and more
- Price volatility index: Tomato (Very High, 5×) · Onion (Very High, 10×) · Wheat (Low, MSP-supported)

</details>

<details>
<summary><b>🏛️ Government Scheme Matching</b></summary>

| Scheme | Benefit |
|--------|---------|
| PMFBY | Crop insurance — ₹15,500 Cr budget |
| PM-KISAN | ₹6,000/year income support |
| KCC | 4% interest credit |
| Soil Health Card | Free soil testing |
| PMKSY | 55% irrigation subsidy |
| e-NAM | Access to 1,361 mandis |
| PM Kisan Maandhan | ₹3,000/month pension |
| AIF | ₹1 Lakh Cr infrastructure fund |
| PKVY | Organic farming subsidy |
| SMAM | Equipment subsidy |

</details>

<details>
<summary><b>🧠 Persistent Farm Memory</b></summary>

Firestore collections: `farm_memory` · `conversation_history` · `disease_history` · `execution_logs` · `dlq_events`

- Schema version 2
- In-memory cache with TTL (30s traces, 120s history)
- Dead Letter Queue for failed batch persists (max 3 retries)
- EventBus batched flush: 2 seconds or 50 events

</details>

<details>
<summary><b>🌐 Multilingual Interaction</b></summary>

- **Speech recognition:** 11 Indian languages with auto-detect (`hi-IN → en-IN → ta-IN → …`)
- **TTS:** 12 languages via backend edge-tts + 10 via browser fallback, 29 voices
- **Translation agent:** Tamil ↔ English, 90% fixed confidence
- Language detection via Unicode script analysis

</details>

---

## 🛠️ Technology Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 16.2.9 | Framework (App Router, static export) |
| TypeScript | 5.x | Type safety (strict mode) |
| Tailwind CSS | 4.x | Styling |
| Framer Motion | 12.42.0 | Animations |
| shadcn/ui | Latest | UI components (20+) |
| Recharts | 3.9.1 | Charts (pie, bar, line) |
| Firebase Client | 12.15.0 | Authentication |

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.115+ | Web framework |
| Python | 3.10+ | Runtime |
| google-genai | 1.0+ | Gemini SDK |
| firebase-admin | 6.0+ | Auth + Firestore |
| ultralytics | 8.3+ | YOLO11 |
| edge-tts | 6.0+ | Text-to-Speech (no key needed) |
| huggingface_hub | 0.27+ | HuggingFace Inference |
| tenacity | 8.0+ | Retry logic |
| g4f | 0.4+ | Fallback LLM |
| **mcp** | **1.28+** | **Model Context Protocol server** |

### AI & ML Models

| Model | Provider | Purpose |
|-------|----------|---------|
| gemini-2.5-flash | Google | Primary LLM — orchestration + agents |
| gemma-4-26b-a4b-it | OpenRouter | Primary vision analysis (free) |
| gemini-2.0-flash | Google | Vision fallback |
| Qwen2.5-VL-7B-Instruct | HuggingFace | Expert vision analysis |
| YOLO11n | Ultralytics (local) | 17 disease + 10 waste class detection |
| crop_leaf_diseases_vit | HuggingFace | Plant disease classification |
| plant-disease-mobilenetv2 | HuggingFace | Plant disease classification |
| Augmented-Waste-Classifier-SigLIP2 | HuggingFace | Waste classification |
| GPT-4o (g4f) | Free tier | Gemini exhaustion fallback |

---

## 🔌 MCP Server

SEED AI exposes its entire knowledge base as a standalone **Model Context Protocol** server — any MCP-compatible client can discover and call these tools.

```bash
# stdio mode (CLI)
python mcp_server/server.py --transport stdio

# HTTP mode (SSE)
python mcp_server/server.py --transport sse
```

| Tool | Knowledge Base Source |
|------|-----------------------|
| `get_disease_info` | diseases.json |
| `get_treatment_plan` | treatments.json |
| `get_crop_guide` | crops.json |
| `find_government_schemes` | government_schemes.json |
| `get_pest_management` | pests.json |
| `get_soil_guide` | soil_health.json |
| `get_market_prices` | market_intelligence.json |
| `knowledge_base_categories` | All KB files |

**Resources:** `seedai://diseases/list` · `seedai://crops/list` · `seedai://schemes/list` · `seedai://pests/list`

> Built with FastMCP (`mcp` Python SDK) — entire server in under 150 lines.

---

## 📡 Real-Time Streaming

Every agent step is streamed live to the frontend via **Server-Sent Events (SSE)** — authenticated with Firebase Bearer tokens, delivered through a fetch-based ReadableStream (since `EventSource` doesn't support headers).

| Event Type | Phase | Payload |
|------------|-------|---------|
| `event` | Running | `{ agent, message, latency_ms, tools_used, status, timestamp }` |
| `pre_result` | Fusion ready | `{ message: "Tier-1 fusion ready…" }` |
| `result` | Complete | `{ summary, recommended_actions, confidence, agent_results, execution_metadata }` |
| `error` | Error | `{ message, degraded_result }` |

Frontend circuit breaker: 4 failover URLs · `FAILURE_THRESHOLD = 3` · 30s cooldown · retry on 429 + 5xx.

---

## 🔒 Security & Isolation

Production-grade multi-tenant isolation enforced at every layer:

```
┌─ Firebase Auth ──────────────────────────────────────────────┐
│  Anonymous · Email/Password · Google OAuth                    │
│  Token refresh every 50 minutes (background interval)         │
└──────────────────────────────────────────────────────────────┘
┌─ Backend Authorization ──────────────────────────────────────┐
│  require_auth FastAPI dependency — verifies every Bearer token│
│  SHA-256 token cache (5-min TTL, 100-entry, oldest-eviction)  │
│  /orchestrate overrides client user_id with verified UID      │
└──────────────────────────────────────────────────────────────┘
┌─ In-Memory Isolation ────────────────────────────────────────┐
│  AgentEvaluator.trace_data: Dict[user_id, list]               │
│  All traces, stats, and reports scoped per authenticated user │
└──────────────────────────────────────────────────────────────┘
┌─ Firestore Security Rules ───────────────────────────────────┐
│  farm_memory/{userId}: request.auth.uid == userId             │
│  conversation_history, execution_logs, disease_history:       │
│    resource.data.user_id == request.auth.uid                  │
│  Default deny for all unlisted collections                    │
└──────────────────────────────────────────────────────────────┘
┌─ Additional Hardening ───────────────────────────────────────┐
│  Prompt injection sanitization (5 patterns, BaseAgent)        │
│  No secrets in source — .gitignore blocks all key files       │
│  SSE auth headers (never query params — no URL exposure)      │
└──────────────────────────────────────────────────────────────┘
```

---

## ⚡ Performance

| Layer | Strategy | Value |
|-------|----------|-------|
| Orchestrator cache | md5 hash, LRU | 300s TTL, 100 entries |
| Memory agent cache | TTL + prefix invalidation | 30s traces / 120s history |
| Token verification | SHA-256 + LRU | 5-min TTL, 100 entries |
| TTS audio | SHA-256 file cache | Auto-cleanup by max age |
| React components | `React.memo` on 5 components | `useMemo` + `useCallback` |
| SSE event flusher | Debounced state updates | 50ms throttle |
| Gemini key pool | Round-robin, cooldown, g4f fallback | Up to 11 keys |
| Agent timeout | Per-agent asyncio | 60 seconds |
| Agent retries | Transient error recovery | 2 attempts |

---

## 🎯 Example Scenario

**Input from farmer:**
> *"I have three acres of tomatoes. Budget ₹4,000. Rain expected tomorrow."*
> *(+ diseased leaf image uploaded)*

**SEED AI's autonomous pipeline:**

```
1. 🔬 Vision Agent       → Early Blight (Alternaria solani), 94% confidence, Moderate (30%)
2. 🌤️ Weather Agent      → Rain probability 70% tomorrow, 28°C, 85% humidity
3. 🌿 Crop Knowledge     → Protocols: Mancozeb ₹280, Chlorothalonil ₹450, Azoxystrobin ₹800
4. 💰 Budget Agent       → Mancozeb ₹280 recommended — ₹2,950 remaining from ₹4,000 budget
5. 📈 Market Agent       → Tomato ₹35/kg stable — wait 2 weeks for harvest
6. 🏛️ Schemes Agent      → Eligible: PMFBY, PM-KISAN, KCC, Soil Health Card
7. 📅 Task Planning      → 7-day schedule generated
8. 🛡️ Guardrails         → Rain 70% > 60% → spray postponed from Day 1 to Day 2
9. 🧩 Fusion             → Combined plan, 88% confidence (High)
10. 🧠 Memory            → Disease event, treatment, budget usage saved to Firestore
11. 📡 SSE Stream        → Every step shown live with agent latencies
```

**Result: One actionable plan in under 15 seconds.**

---

## 🖥️ User Interface

```
┌─ SEED AI Command Center ──────────────────────────────────────┐
│  [🎤 Mic]  [📷 Upload Image]  [Query Input...]  [Run →]       │
│  Mode: [Auto ▼]   Agent: [Any ▼]                              │
│  Presets: [Tomato Blight] [Rice Irrigation] [Budget Plan]     │
├───────────────────────────────────────────────────────────────┤
│  ┌─ Swarm Visualization ──┐  ┌─ Live Event Log ─────────────┐ │
│  │   ●──●──●──●──●        │  │  ✓ Vision Agent     1.2s    │ │
│  │   │  │  │  │  │        │  │  ⏳ Weather Agent    0.8s    │ │
│  │   ●──●──●──●──●        │  │  ✓ Crop K. Agent    1.5s    │ │
│  │   │  │  │  │  │        │  │  ✓ Budget Agent     0.6s    │ │
│  │   ●──●──●──●──●        │  │  ✓ Market Agent     1.1s    │ │
│  └────────────────────────┘  └──────────────────────────────┘ │
│                                                               │
│  ┌─ Results ───────────────────────────────────────────────┐  │
│  │  [Action Plan] [Timeline] [Weather] [Soil] [Markets]    │  │
│  │                                                          │  │
│  │  🌿 Disease: Early Blight (94% confidence)              │  │
│  │  ⚠️  Severity: Moderate — 30% affected area             │  │
│  │  💊 Treatment: Mancozeb 75% WP — ₹280 / ₹4,000 budget  │  │
│  │  🌧️  Rain alert: 70% — spray postponed to Day 2        │  │
│  │                                                          │  │
│  │  📅 7-Day Schedule:                                      │  │
│  │  ☐ Day 1: Inspect field + prepare spray solution        │  │
│  │  ☐ Day 2: Apply Mancozeb (morning, post-rain)           │  │
│  │  ☐ Day 3: Monitor for improvement                       │  │
│  │  ☐ Day 5: Drip irrigation (morning)                     │  │
│  │  ☐ Day 7: Upload follow-up leaf image                   │  │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Responsible AI

SEED AI is built with trustworthy AI as a first-class principle — not an afterthought.

- **Confidence scores** on every recommendation: High (90–100%) · Medium (75–89%) · Low (<75%)
- **Reasoning fields** in every AgentResult and the FusionResult
- **Evidence sources** cited — `evidence_sources` links to specific data used
- **Guardrails before output** — rain-check and budget ceiling modify results, not just warn
- **Reflection after fusion** — Gemini self-critique catches safety issues and revises outputs
- **No hallucinated statistics** — unsupported claims are explicitly avoided
- **Expert consultation encouraged** for severe disease conditions
- **Prompt injection protection** — 5 sanitization patterns applied in BaseAgent across all agents
- **Privacy enforced end-to-end** — Firebase Auth + Firestore rules + backend UID verification
- **No secrets in code** — `.gitignore` blocks all credential files from source control

---

## 📊 Impact Metrics

<div align="center">

| Metric | Target |
|--------|--------|
| ⚡ Disease Analysis Time | **< 10 seconds** |
| 📋 Personalized Plan Generation | **< 15 seconds** |
| 🌐 Supported Languages (Speech / TTS) | **11 / 12** |
| 🔧 External Tool Calls per Query | **6 – 10** |
| 🧠 Farm Memory | **Persistent (Firestore, schema v2)** |
| 🤖 AI Agents | **15 (13 specialist + memory + translate)** |
| 🔁 Concurrent Tier 1 Agents | **Up to 8** |
| 🌾 Supported Crops | **8** |
| 🔬 Supported Diseases | **15** |
| 🐛 Supported Pests | **6** |
| 🌱 Supported Soil Types | **5** |
| 🏛️ Government Schemes | **10** |
| 💊 Treatment Options in KB | **40+** |
| 📈 Crop MSPs Tracked | **26** |
| 🔑 LLM Keys (round-robin pool) | **Up to 11** |
| ⏱️ Agent Timeout | **60 seconds** |
| 🔄 Retry Attempts | **2 per agent** |

</div>

---

<div align="center">

## 🌱 Built With Purpose

*It demonstrates that modern AI agents — given planning, reasoning, tool use, memory, reflection, guardrails, and collaboration — can move beyond answering questions to making real decisions that improve lives.*

<br/>

**Agriculture is one of humanity's oldest challenges.**
**Autonomous AI agents may be one of its most powerful new tools.**

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
