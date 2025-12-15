"""Tests for the LangGraph pipeline."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.pipeline.graph import AIPipeline, PipelineState
from src.agents.router import RouterAgent, RouterDecision, QueryType


class TestRouterAgent:
    """Tests for RouterAgent class."""
    
    @pytest.fixture
    def mock_router(self):
        """Create a mock router agent."""
        with patch("src.agents.router.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                gemini_model="gemini-1.5-flash",
                google_api_key="test_key"
            )
            with patch("src.agents.router.ChatGoogleGenerativeAI") as mock_llm:
                # Create a mock chain that returns RouterDecision
                mock_chain = Mock()
                mock_chain.invoke.return_value = RouterDecision(
                    query_type="weather",
                    reasoning="Query mentions weather",
                    extracted_entity="London"
                )
                
                router = RouterAgent.__new__(RouterAgent)
                router._chain = mock_chain
                router._llm = mock_llm
                return router
    
    def test_route_weather_query(self, mock_router):
        """Test routing a weather query."""
        decision = mock_router.route("What's the weather in London?")
        
        assert decision.query_type == "weather"
        assert decision.extracted_entity == "London"
    
    def test_get_query_type(self, mock_router):
        """Test getting query type enum."""
        query_type = mock_router.get_query_type("Weather in Paris")
        
        assert query_type == QueryType.WEATHER


class TestRouterDecision:
    """Tests for RouterDecision model."""
    
    def test_router_decision_creation(self):
        """Test creating RouterDecision."""
        decision = RouterDecision(
            query_type="pdf",
            reasoning="Query about document",
            extracted_entity="summary"
        )
        
        assert decision.query_type == "pdf"
        assert decision.reasoning == "Query about document"
    
    def test_router_decision_default_entity(self):
        """Test RouterDecision with default entity."""
        decision = RouterDecision(
            query_type="unknown",
            reasoning="Unclear query"
        )
        
        assert decision.extracted_entity == ""


class TestPipelineState:
    """Tests for PipelineState TypedDict."""
    
    def test_pipeline_state_creation(self):
        """Test creating a PipelineState."""
        state: PipelineState = {
            "query": "Test query",
            "query_type": "weather",
            "extracted_entity": "London",
            "routing_reasoning": "Weather query",
            "response": "It's sunny",
            "sources": [],
            "success": True,
            "agent_used": "weather"
        }
        
        assert state["query"] == "Test query"
        assert state["success"] == True


class TestAIPipeline:
    """Tests for AIPipeline class."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline."""
        with patch("src.pipeline.graph.RouterAgent") as mock_router_cls:
            with patch("src.pipeline.graph.WeatherAgent") as mock_weather_cls:
                with patch("src.pipeline.graph.RAGAgent") as mock_rag_cls:
                    # Setup mocks
                    mock_router = Mock()
                    mock_router.route.return_value = RouterDecision(
                        query_type="weather",
                        reasoning="Weather query",
                        extracted_entity="London"
                    )
                    mock_router_cls.return_value = mock_router
                    
                    mock_weather = Mock()
                    mock_weather.process.return_value = {
                        "response": "It's 20°C in London",
                        "success": True,
                        "city": "London"
                    }
                    mock_weather_cls.return_value = mock_weather
                    
                    mock_rag = Mock()
                    mock_rag.process.return_value = {
                        "response": "Document summary",
                        "sources": ["doc.pdf"],
                        "success": True
                    }
                    mock_rag_cls.return_value = mock_rag
                    
                    pipeline = AIPipeline()
                    pipeline._router = mock_router
                    pipeline._weather_agent = mock_weather
                    pipeline._rag_agent = mock_rag
                    
                    return pipeline
    
    def test_router_node(self, mock_pipeline):
        """Test router node processing."""
        state = {"query": "Weather in London"}
        result = mock_pipeline._router_node(state)
        
        assert result["query_type"] == "weather"
        assert result["extracted_entity"] == "London"
    
    def test_route_query_weather(self, mock_pipeline):
        """Test query routing to weather."""
        state = {"query_type": "weather"}
        next_node = mock_pipeline._route_query(state)
        
        assert next_node == "weather"
    
    def test_route_query_pdf(self, mock_pipeline):
        """Test query routing to PDF."""
        state = {"query_type": "pdf"}
        next_node = mock_pipeline._route_query(state)
        
        assert next_node == "pdf"
    
    def test_route_query_unknown(self, mock_pipeline):
        """Test query routing to unknown."""
        state = {"query_type": "something_else"}
        next_node = mock_pipeline._route_query(state)
        
        assert next_node == "unknown"
    
    def test_weather_node(self, mock_pipeline):
        """Test weather agent node."""
        state = {
            "query": "Weather in London",
            "extracted_entity": "London"
        }
        result = mock_pipeline._weather_node(state)
        
        assert result["agent_used"] == "weather"
        assert "20°C" in result["response"]
    
    def test_rag_node(self, mock_pipeline):
        """Test RAG agent node."""
        state = {"query": "Summarize the document"}
        result = mock_pipeline._rag_node(state)
        
        assert result["agent_used"] == "rag"
        assert "doc.pdf" in result["sources"]
    
    def test_unknown_node(self, mock_pipeline):
        """Test unknown handler node."""
        state = {"query": "Random unclear query"}
        result = mock_pipeline._unknown_node(state)
        
        assert result["agent_used"] == "unknown"
        assert result["success"] == False
        assert "Weather queries" in result["response"]
    
    def test_get_graph_diagram(self, mock_pipeline):
        """Test getting mermaid diagram."""
        diagram = mock_pipeline.get_graph_diagram()
        
        assert "mermaid" in diagram
        assert "Router Node" in diagram
        assert "Weather Agent" in diagram


class TestPipelineIntegration:
    """Integration tests for the pipeline."""
    
    def test_invoke_returns_valid_state(self):
        """Test that invoke returns a valid state structure."""
        with patch("src.pipeline.graph.RouterAgent") as mock_router_cls:
            with patch("src.pipeline.graph.WeatherAgent") as mock_weather_cls:
                with patch("src.pipeline.graph.RAGAgent") as mock_rag_cls:
                    # Minimal mocks
                    mock_router = Mock()
                    mock_router.route.return_value = RouterDecision(
                        query_type="unknown",
                        reasoning="Test",
                        extracted_entity=""
                    )
                    mock_router_cls.return_value = mock_router
                    mock_weather_cls.return_value = Mock()
                    mock_rag_cls.return_value = Mock()
                    
                    pipeline = AIPipeline()
                    result = pipeline.invoke("Hello")
                    
                    # Check result structure
                    assert "query" in result
                    assert "response" in result
                    assert "agent_used" in result
