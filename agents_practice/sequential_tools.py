
import requests

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

load_dotenv()

@tool
def get_city_info(city: str) -> str:
    """Get general information about a city."""

    url = "https://en.wikipedia.org/w/api.php"
    response = requests.get(
        url,
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

    return summary_response.json().get("extract", "No summary available")

@tool
def get_weather(city: str) -> str:
    """Get current weather information for a city."""

    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_response = requests.get(
        geo_url,
        params={"name": city, "count": 1},
        timeout=10
    )
    geo_response.raise_for_status()
    geo_data = geo_response.json()

    if not geo_data.get("results"):
        return f"Could not find {city}"

    location = geo_data["results"][0]
    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_response = requests.get(
        weather_url,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature"
            ),
            "timezone": "auto"
        },
        timeout=10
    )

    weather_response.raise_for_status()
    current = weather_response.json()["current"]
    return (
        f"Temperature: {current['temperature_2m']}°C\n"
        f"Feels like: {current['apparent_temperature']}°C\n"
        f"Humidity: {current['relative_humidity_2m']}%"
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
question = input("Ask me something: ")

response = llm.invoke(
    f"Identify the destination city from this request. "
    f"Return only the city name. If no city is mentioned, "
    f"say NOT_FOUND.\nRequest: {question}"
)

city = response.content.strip()

if city.upper() == "NOT_FOUND":
    print("Please mention a destination city.")
    raise SystemExit

print("\nDestination:", city)

print("\nCalling first tool: get_city_info")

city_info = get_city_info.invoke({"city": city})

print("\nCity Information:")
print(city_info)

print("\nCalling second tool: get_weather")

weather_info = get_weather.invoke({"city": city})

print("\nWeather Information:")
print(weather_info)

final_response = llm.invoke(
    f"""
    Answer the user's travel request using the information below.
    User Request: {question}
    City Information:
    {city_info}
    Weather Information:
    {weather_info}
    Give a useful, clear response.
    """
)

print("\nFinal Answer:")
print(final_response.content)
