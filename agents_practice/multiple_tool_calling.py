
import os
import requests

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get the current weather conditions for a given city."""

    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_response = requests.get(
            geo_url,
            params={"name": city, "count": 1},
            timeout=10
        )
        geo_response.raise_for_status()

        geo_data = geo_response.json()
        print(f"Geocoding response: {geo_data}")

        if not geo_data.get("results"):
            return f"Could not find the city: {city}"

        location = geo_data["results"][0]
        print(f"Location found: {location['name']}, {location.get('country', 'Unknown')}")

        latitude = location["latitude"]
        longitude = location["longitude"]
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
                    "wind_speed_10m"
                ),
                "timezone": "auto"
            },
            timeout=10
        )
        weather_response.raise_for_status()

        weather_data = weather_response.json()
        current = weather_data["current"]

        return (
            f"City: {location['name']}\n"
            f"Country: {location.get('country', 'Unknown')}\n"
            f"Temperature: {current['temperature_2m']}°C\n"
            f"Feels like: {current['apparent_temperature']}°C\n"
            f"Humidity: {current['relative_humidity_2m']}%\n"
            f"Wind speed: {current['wind_speed_10m']} km/h"
        )

    except requests.RequestException as e:
        return f"Weather API error: {e}"
    except (KeyError, ValueError) as e:
        return f"Could not process weather data: {e}"

@tool
def get_city_info(city: str) -> str:
    """Get general information about a city using Wikipedia."""

    try:
        search_url = "https://en.wikipedia.org/w/api.php"

        response = requests.get(
            search_url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": city,
                "format": "json"
            },
            headers={"User-Agent": "LearningAgent/1.0"},
            timeout=10
        )
        response.raise_for_status()
        results = response.json()["query"]["search"]
        if not results:
            return f"No information found for {city}"
        title = results[0]["title"]
        summary_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + requests.utils.quote(title.replace(" ", "_"), safe="")
        )
        summary_response = requests.get(
            summary_url,
            headers={"User-Agent": "LearningAgent/1.0"},
            timeout=10
        )
        summary_response.raise_for_status()
        summary = summary_response.json()
        return summary.get("extract", f"No summary available for {city}")
    except requests.RequestException as e:
        return f"City information API error: {e}"

@tool
def get_country_info(country: str) -> str:
    """Get general information about a country using Wikipedia."""

    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        response = requests.get(
            search_url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": country,
                "format": "json"
            },
            headers={"User-Agent": "LearningAgent/1.0"},
            timeout=10
        )
        response.raise_for_status()
        results = response.json()["query"]["search"]
        if not results:
            return f"No information found for {country}"
        title = results[0]["title"]
        summary_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + requests.utils.quote(title.replace(" ", "_"), safe="")
        )
        summary_response = requests.get(
            summary_url,
            headers={"User-Agent": "LearningAgent/1.0"},
            timeout=10
        )
        summary_response.raise_for_status()
        summary = summary_response.json()
        return summary.get("extract", f"No summary available for {country}")
    except requests.RequestException as e:
        return f"Country information API error: {e}"

tools = [
    get_weather,
    get_city_info,
    get_country_info
]

tools_by_name = {
    tool.name: tool for tool in tools
}

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)
question = input("Ask me something: ")
response = llm_with_tools.invoke(question)
if response.tool_calls:
    tool_messages = []
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        print(f"Selected tool: {tool_name}")
        tool_args = tool_call["args"]
        print(f"Tool arguments: {tool_args}")
        tool_id = tool_call["id"]

        print(f"\nCalling tool: {tool_name}")
        print(f"Arguments: {tool_args}")

        selected_tool = tools_by_name[tool_name]
        tool_result = selected_tool.invoke(tool_args)
        print(f"Tool result: {tool_result}")
        tool_messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_id
            )
        )
    final_response = llm_with_tools.invoke(
        [("human", question), response, *tool_messages]
    )
    print("\nFinal Answer:")
    print(final_response.content)
else:
    print("\nAnswer:")
    print(response.content)
