"""Tests for the weather service."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx

from src.services.weather import WeatherService, WeatherData


class TestWeatherData:
    """Tests for WeatherData dataclass."""
    
    def test_weather_data_creation(self):
        """Test creating a WeatherData instance."""
        weather = WeatherData(
            city="London",
            country="GB",
            temperature=15.5,
            feels_like=14.0,
            humidity=75,
            description="light rain",
            wind_speed=5.2,
            icon="10d"
        )
        
        assert weather.city == "London"
        assert weather.country == "GB"
        assert weather.temperature == 15.5
        assert weather.humidity == 75


class TestWeatherService:
    """Tests for WeatherService class."""
    
    @pytest.fixture
    def mock_response(self):
        """Create a mock API response."""
        return {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {
                "temp": 15.5,
                "feels_like": 14.0,
                "humidity": 75
            },
            "weather": [{"description": "light rain", "icon": "10d"}],
            "wind": {"speed": 5.2}
        }
    
    @pytest.fixture
    def weather_service(self):
        """Create a WeatherService with a mock API key."""
        with patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "test_key"}):
            return WeatherService(api_key="test_key")
    
    def test_initialization_with_api_key(self):
        """Test service initialization with explicit API key."""
        service = WeatherService(api_key="my_test_key")
        assert service.api_key == "my_test_key"
    
    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch("src.services.weather.get_settings") as mock_settings:
            mock_settings.return_value = Mock(openweathermap_api_key="")
            with pytest.raises(ValueError, match="API key is required"):
                WeatherService()
    
    @patch("httpx.Client.get")
    def test_get_current_weather_success(self, mock_get, weather_service, mock_response):
        """Test successful weather fetch."""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: mock_response
        )
        
        weather = weather_service.get_current_weather("London")
        
        assert weather.city == "London"
        assert weather.country == "GB"
        assert weather.temperature == 15.5
        assert weather.description == "light rain"
    
    @patch("httpx.Client.get")
    def test_get_current_weather_city_not_found(self, mock_get, weather_service):
        """Test handling of city not found."""
        mock_get.return_value = Mock(status_code=404)
        
        with pytest.raises(ValueError, match="not found"):
            weather_service.get_current_weather("InvalidCity123")
    
    @patch("httpx.Client.get")
    def test_get_forecast_success(self, mock_get, weather_service):
        """Test successful forecast fetch."""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"list": [{"temp": 15}, {"temp": 16}]}
        )
        
        forecast = weather_service.get_forecast("London", days=2)
        
        assert len(forecast) == 2
    
    def test_format_weather_report(self, weather_service):
        """Test weather report formatting."""
        weather = WeatherData(
            city="Paris",
            country="FR",
            temperature=20.0,
            feels_like=19.0,
            humidity=60,
            description="clear sky",
            wind_speed=3.0,
            icon="01d"
        )
        
        report = weather_service.format_weather_report(weather)
        
        assert "Paris" in report
        assert "20.0°C" in report
        assert "60%" in report
        assert "Clear Sky" in report
    
    def test_context_manager(self, weather_service):
        """Test context manager functionality."""
        with weather_service as service:
            assert service is not None
        # Should not raise after exiting context
