"""Weather agent for handling weather-related queries."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import get_settings
from src.services.weather import WeatherService, WeatherData


class WeatherAgent:
    """Agent that handles weather-related queries."""
    
    WEATHER_PROMPT = """You are a helpful weather assistant. Based on the weather data provided, 
answer the user's question in a friendly and informative way.

Weather Data:
{weather_data}

User Question: {question}

Provide a natural, conversational response that answers the question using the weather data.
Include relevant emojis to make the response engaging.
If the question asks for something not in the data, politely explain what information is available."""

    def __init__(self):
        """Initialize the weather agent."""
        settings = get_settings()
        
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.7
        )
        
        self._weather_service = WeatherService()
        self._prompt = ChatPromptTemplate.from_template(self.WEATHER_PROMPT)
        self._chain = self._prompt | self._llm | StrOutputParser()
    
    def get_weather(self, city: str) -> WeatherData:
        """Get weather data for a city.
        
        Args:
            city: City name to get weather for
            
        Returns:
            WeatherData object
        """
        return self._weather_service.get_current_weather(city)
    
    def answer_weather_query(self, question: str, city: str) -> str:
        """Answer a weather-related question.
        
        Args:
            question: User's weather question
            city: City to get weather for
            
        Returns:
            Natural language response about the weather
        """
        try:
            weather = self.get_weather(city)
            weather_report = self._weather_service.format_weather_report(weather)
            
            response = self._chain.invoke({
                "weather_data": weather_report,
                "question": question
            })
            
            return response
            
        except ValueError as e:
            return f"I couldn't find weather data for '{city}'. Please check the city name and try again. Error: {str(e)}"
        except Exception as e:
            return f"Sorry, I encountered an error while fetching weather data: {str(e)}"
    
    def process(self, query: str, extracted_city: str = "") -> dict:
        """Process a weather query end-to-end.
        
        Args:
            query: User's original query
            extracted_city: Pre-extracted city name (optional)
            
        Returns:
            Dictionary with response and metadata
        """
        city = extracted_city.strip()
        
        if not city:
            return {
                "response": "I need a city name to fetch weather information. Please specify a location like 'What's the weather in London?'",
                "success": False,
                "city": None
            }
        
        response = self.answer_weather_query(query, city)
        
        return {
            "response": response,
            "success": "couldn't find" not in response.lower() and "error" not in response.lower(),
            "city": city
        }
