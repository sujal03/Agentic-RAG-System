"""Router agent for query classification."""
from enum import Enum
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from src.config import get_settings


class QueryType(str, Enum):
    """Types of queries the system can handle."""
    WEATHER = "weather"
    PDF = "pdf"
    UNKNOWN = "unknown"


class RouterDecision(BaseModel):
    """Structured output for routing decision."""
    query_type: Literal["weather", "pdf", "unknown"] = Field(
        description="The type of query: 'weather' for weather-related questions, 'pdf' for document-related questions, 'unknown' for unclear queries"
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )
    extracted_entity: str = Field(
        default="",
        description="Extracted entity (city name for weather, topic for PDF)"
    )


class RouterAgent:
    """Agent that routes queries to appropriate handlers."""
    
    ROUTER_PROMPT = """You are a query router that classifies user questions into categories.

Analyze the user's question and determine if it's asking about:
1. **weather** - Questions about current weather, temperature, forecast, climate conditions for a location
   Examples: "What's the weather in London?", "Is it raining in Tokyo?", "Temperature in Paris"
2. **pdf** - Questions about a document, asking to find information, summarize content, or answer from uploaded files
   Examples: "What does the document say about X?", "Summarize the PDF", "Find information about Y in the file"
3. **unknown** - Questions that don't fit either category or are unclear

For weather queries, extract the city/location name.
For PDF queries, extract the main topic or keywords.

User Query: {query}

Respond with:
- query_type: "weather", "pdf", or "unknown"
- reasoning: Brief explanation
- extracted_entity: City name (for weather) or topic (for pdf)"""

    def __init__(self):
        """Initialize the router agent."""
        settings = get_settings()
        
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0
        )
        
        self._prompt = ChatPromptTemplate.from_template(self.ROUTER_PROMPT)
        
        # Use structured output
        self._chain = self._prompt | self._llm.with_structured_output(RouterDecision)
    
    def route(self, query: str) -> RouterDecision:
        """Route a query to the appropriate handler.
        
        Args:
            query: User's input query
            
        Returns:
            RouterDecision with query type and extracted entity
        """
        return self._chain.invoke({"query": query})
    
    def get_query_type(self, query: str) -> QueryType:
        """Get just the query type for a query.
        
        Args:
            query: User's input query
            
        Returns:
            QueryType enum value
        """
        decision = self.route(query)
        return QueryType(decision.query_type)
