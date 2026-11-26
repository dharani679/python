
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage
import requests

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get the current weather information for a given city."""

    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_response = requests.get(
            geo_url,
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=15
        )

        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"City '{city}' was not found."

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]
        city_name = location["name"]
        country = location.get("country", "Unknown")

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "precipitation,"
                    "wind_speed_10m,"
                    "weather_code"
                ),
                "timezone": "auto"
            },
            timeout=15
        )

        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data["current"]

        return (
            f"Weather in {city_name}, {country}:\n"
            f"Temperature: {current['temperature_2m']}°C\n"
            f"Feels Like: {current['apparent_temperature']}°C\n"
            f"Humidity: {current['relative_humidity_2m']}%\n"
            f"Wind Speed: {current['wind_speed_10m']} km/h\n"
            f"Precipitation: {current['precipitation']} mm\n"
            f"Weather Code: {current['weather_code']}\n"
            f"Observation Time: {current['time']}"
        )

    except requests.RequestException as e:
        return f"Weather API request failed: {str(e)}"

    except (KeyError, ValueError) as e:
        return f"Could not process weather data: {str(e)}"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)
question = input("Ask your question: ")
response = llm_with_tools.invoke(question)
if response.tool_calls:

    tool_messages = []

    for tool_call in response.tool_calls:

        print("\nSelected Tool:", tool_call["name"])
        print("Tool Arguments:", tool_call["args"])
        selected_tool = {
            "get_weather": get_weather
        }[tool_call["name"]]

        result = selected_tool.invoke(tool_call["args"])

        print("\nTool Result:\n", result)

        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )

        tool_messages.append(tool_message)

    final_response = llm_with_tools.invoke(
        [
            ("human", question),
            response,
            *tool_messages
        ]
    )
    print("\nFinal Answer:", final_response.content)
else:
    print("\nAnswer:", response.content)