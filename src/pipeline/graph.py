"""LangGraph pipeline for the AI agent system."""
from typing import TypedDict, Literal, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from src.agents.router import RouterAgent, QueryType
from src.agents.weather_agent import WeatherAgent
from src.agents.rag_agent import RAGAgent


class PipelineState(TypedDict):
    """State for the LangGraph pipeline."""
    # Input
    query: str
    
    # Routing
    query_type: str
    extracted_entity: str
    routing_reasoning: str
    
    # Output
    response: str
    sources: list[str]
    success: bool
    
    # Metadata
    agent_used: str


class AIPipeline:
    """LangGraph-based AI pipeline for weather and PDF queries."""
    
    def __init__(self, rag_agent: Optional[RAGAgent] = None):
        """Initialize the pipeline.
        
        Args:
            rag_agent: Optional RAGAgent instance (for sharing indexed documents)
        """
        self._router = RouterAgent()
        self._weather_agent = WeatherAgent()
        self._rag_agent = rag_agent or RAGAgent()
        
        # Build the graph
        self._graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.
        
        Returns:
            Compiled StateGraph
        """
        # Create graph with state schema
        builder = StateGraph(PipelineState)
        
        # Add nodes
        builder.add_node("router", self._router_node)
        builder.add_node("weather_agent", self._weather_node)
        builder.add_node("rag_agent", self._rag_node)
        builder.add_node("unknown_handler", self._unknown_node)
        
        # Add edges
        builder.add_edge(START, "router")
        
        # Conditional routing after router node
        builder.add_conditional_edges(
            "router",
            self._route_query,
            {
                "weather": "weather_agent",
                "pdf": "rag_agent",
                "unknown": "unknown_handler"
            }
        )
        
        # All agents end the graph
        builder.add_edge("weather_agent", END)
        builder.add_edge("rag_agent", END)
        builder.add_edge("unknown_handler", END)
        
        return builder.compile()
    
    def _router_node(self, state: PipelineState) -> dict:
        """Router node that classifies the query.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with routing information
        """
        decision = self._router.route(state["query"])
        
        return {
            "query_type": decision.query_type,
            "extracted_entity": decision.extracted_entity,
            "routing_reasoning": decision.reasoning
        }
    
    def _route_query(self, state: PipelineState) -> Literal["weather", "pdf", "unknown"]:
        """Determine which agent to route to.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Name of the next node
        """
        query_type = state.get("query_type", "unknown")
        
        if query_type == "weather":
            return "weather"
        elif query_type == "pdf":
            return "pdf"
        else:
            return "unknown"
    
    def _weather_node(self, state: PipelineState) -> dict:
        """Weather agent node.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with weather response
        """
        result = self._weather_agent.process(
            query=state["query"],
            extracted_city=state.get("extracted_entity", "")
        )
        
        return {
            "response": result["response"],
            "success": result["success"],
            "sources": [],
            "agent_used": "weather"
        }
    
    def _rag_node(self, state: PipelineState) -> dict:
        """RAG agent node.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with RAG response
        """
        result = self._rag_agent.process(state["query"])
        
        return {
            "response": result["response"],
            "success": result["success"],
            "sources": result.get("sources", []),
            "agent_used": "rag"
        }
    
    def _unknown_node(self, state: PipelineState) -> dict:
        """Handle unknown query types.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with helpful message
        """
        return {
            "response": (
                "I'm not sure how to handle that question. I can help you with:\n\n"
                "🌤️ **Weather queries**: Ask about weather in any city\n"
                "   Example: 'What's the weather in Tokyo?'\n\n"
                "📄 **Document questions**: Ask about uploaded PDF content\n"
                "   Example: 'What does the document say about X?'\n\n"
                "Please try rephrasing your question!"
            ),
            "success": False,
            "sources": [],
            "agent_used": "unknown"
        }
    
    def invoke(self, query: str) -> PipelineState:
        """Run the pipeline on a query.
        
        Args:
            query: User's input query
            
        Returns:
            Final pipeline state with response
        """
        initial_state: PipelineState = {
            "query": query,
            "query_type": "",
            "extracted_entity": "",
            "routing_reasoning": "",
            "response": "",
            "sources": [],
            "success": False,
            "agent_used": ""
        }
        
        return self._graph.invoke(initial_state)
    
    def stream(self, query: str):
        """Stream pipeline execution for debugging.
        
        Args:
            query: User's input query
            
        Yields:
            State updates at each step
        """
        initial_state: PipelineState = {
            "query": query,
            "query_type": "",
            "extracted_entity": "",
            "routing_reasoning": "",
            "response": "",
            "sources": [],
            "success": False,
            "agent_used": ""
        }
        
        for state in self._graph.stream(initial_state):
            yield state
    
    @property
    def rag_agent(self) -> RAGAgent:
        """Get the RAG agent for document loading.
        
        Returns:
            RAGAgent instance
        """
        return self._rag_agent
    
    def get_graph_diagram(self) -> str:
        """Get a Mermaid diagram of the graph.
        
        Returns:
            Mermaid diagram string
        """
        return """
```mermaid
graph TD
    A[Start] --> B[Router Node]
    B -->|weather| C[Weather Agent]
    B -->|pdf| D[RAG Agent]
    B -->|unknown| E[Unknown Handler]
    C --> F[End]
    D --> F
    E --> F
```
"""
