# AI Pipeline with LangChain, LangGraph & LangSmith

An intelligent AI pipeline that uses **LangGraph** to route queries between a real-time weather API and PDF document Q&A using RAG (Retrieval-Augmented Generation), with **LangSmith** evaluation and a **Streamlit** chat interface.

## 🌟 Features

- **🌤️ Weather Queries**: Real-time weather data from OpenWeatherMap API
- **📄 PDF RAG**: Ask questions about uploaded PDF documents
- **🧠 Smart Routing**: LangGraph-powered query classification using Gemini
- **📊 LangSmith Evaluation**: Response quality evaluation with relevance, helpfulness, and coherence metrics
- **💾 Vector Database**: Qdrant for efficient document storage and retrieval
- **💬 Chat Interface**: User-friendly Streamlit UI

## 🏗️ Architecture

```
┌─────────────┐
│   User Query │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│  Router Node │ (Gemini classifies query type)
└──────┬───────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌───────┐ ┌───────┐
│Weather│ │  RAG  │
│ Agent │ │ Agent │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│ Generate Response│ (Gemini LLM)
└─────────────────┘
```

## 📋 Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API Keys:
  - Google AI (Gemini) API key
  - OpenWeatherMap API key
  - LangChain API key (for LangSmith)
  - Qdrant Cloud URL and API key

## 🚀 Setup

### 1. Clone and Install Dependencies

```bash
cd /path/to/project
uv sync
```

Or with pip:
```bash
pip install -e .
```

### 2. Configure Environment Variables

Copy the example file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Google Gemini API
GOOGLE_API_KEY=your_google_api_key

# OpenWeatherMap API
OPENWEATHERMAP_API_KEY=your_openweathermap_key

# LangSmith (for evaluation)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=ai-pipeline

# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 🐳 Docker Setup

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop)

### Quick Start with Docker Compose

1. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   Open `http://localhost:8501` in your browser


## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_weather.py -v
```

## 📁 Project Structure

```
├── src/
│   ├── config.py              # Environment configuration
│   ├── services/
│   │   ├── weather.py         # OpenWeatherMap API client
│   │   ├── embeddings.py      # Google embeddings service
│   │   └── vector_store.py    # Qdrant operations
│   ├── agents/
│   │   ├── router.py          # Query classification
│   │   ├── weather_agent.py   # Weather query handler
│   │   └── rag_agent.py       # PDF RAG handler
│   ├── pipeline/
│   │   └── graph.py           # LangGraph pipeline
│   └── evaluation/
│       └── evaluators.py      # LangSmith evaluators
├── tests/
│   ├── test_weather.py        # Weather service tests
│   ├── test_rag.py            # RAG agent tests
│   └── test_pipeline.py       # Pipeline tests
├── app.py                     # Streamlit application
├── pyproject.toml             # Dependencies
├── requirements.txt           # Pip dependencies
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose configuration
├── .dockerignore              # Docker build exclusions
└── .env.example               # Environment template
```

## 💡 Usage Examples

### Weather Queries
- "What's the weather in London?"
- "Is it raining in Tokyo?"
- "Temperature in New York City"

### Document Queries
1. Upload a PDF using the sidebar
2. Ask questions like:
   - "Summarize this document"
   - "What does the document say about X?"
   - "Find information about Y"

## 📊 LangSmith Evaluation

The pipeline includes custom evaluators for:

- **Relevance**: Is the response relevant to the question?
- **Helpfulness**: Does the response provide actionable information?
- **Coherence**: Is the response well-structured?
- **Faithfulness** (RAG): Is the answer grounded in the context?

View traces at [smith.langchain.com](https://smith.langchain.com)

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemini 1.5 Flash |
| Embeddings | Google text-embedding-004 |
| Agent Framework | LangGraph |
| Vector Database | Qdrant Cloud |
| Evaluation | LangSmith |
| UI | Streamlit |
| Testing | pytest |

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
