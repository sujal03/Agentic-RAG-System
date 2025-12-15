"""Weather service for fetching data from OpenWeatherMap API."""
import httpx
from typing import Optional
from dataclasses import dataclass

from src.config import get_settings


@dataclass
class WeatherData:
    """Weather data container."""
    city: str
    country: str
    temperature: float
    feels_like: float
    humidity: int
    description: str
    wind_speed: float
    icon: str


class WeatherService:
    """Service for fetching weather data from OpenWeatherMap API."""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the weather service.
        
        Args:
            api_key: OpenWeatherMap API key. If not provided, uses env variable.
        """
        settings = get_settings()
        self.api_key = api_key or settings.openweathermap_api_key
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key is required")
        self._client = httpx.Client(timeout=10.0)
    
    def get_current_weather(self, city: str) -> WeatherData:
        """Get current weather for a city.
        
        Args:
            city: City name (e.g., "London" or "London,UK")
            
        Returns:
            WeatherData object with current weather information
            
        Raises:
            httpx.HTTPError: If the API request fails
            ValueError: If the city is not found
        """
        url = f"{self.BASE_URL}/weather"
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        
        response = self._client.get(url, params=params)
        
        if response.status_code == 404:
            raise ValueError(f"City '{city}' not found")
        
        response.raise_for_status()
        data = response.json()
        
        return WeatherData(
            city=data["name"],
            country=data["sys"]["country"],
            temperature=data["main"]["temp"],
            feels_like=data["main"]["feels_like"],
            humidity=data["main"]["humidity"],
            description=data["weather"][0]["description"],
            wind_speed=data["wind"]["speed"],
            icon=data["weather"][0]["icon"]
        )
    
    def get_forecast(self, city: str, days: int = 5) -> list[dict]:
        """Get weather forecast for a city.
        
        Args:
            city: City name
            days: Number of days to forecast (max 5)
            
        Returns:
            List of forecast data dictionaries
        """
        url = f"{self.BASE_URL}/forecast"
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric",
            "cnt": min(days * 8, 40)  # 8 forecasts per day, max 40
        }
        
        response = self._client.get(url, params=params)
        
        if response.status_code == 404:
            raise ValueError(f"City '{city}' not found")
            
        response.raise_for_status()
        data = response.json()
        
        return data["list"]
    
    def format_weather_report(self, weather: WeatherData) -> str:
        """Format weather data into a human-readable report.
        
        Args:
            weather: WeatherData object
            
        Returns:
            Formatted weather report string
        """
        return f"""Current Weather in {weather.city}, {weather.country}:
🌡️ Temperature: {weather.temperature}°C (feels like {weather.feels_like}°C)
💧 Humidity: {weather.humidity}%
🌤️ Conditions: {weather.description.title()}
💨 Wind Speed: {weather.wind_speed} m/s"""
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
