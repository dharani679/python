from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.tools import tool
from langchain_core.messages import ToolMessage


load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get the current weather information for a given city.
    Args:
        city: Name of the city.
    Returns:
        Weather condition and temperature as a string.

    """

    weather_info = f"Weather information for {city}:\n"
    weather_data={
        "hyderabad": "Temperature: 30°C, Humidity: 70%, Condition: Sunny",
        "new york": "Temperature: 25°C, Humidity: 60%, Condition: Cloudy",
        "london": "Temperature: 20°C, Humidity: 80%, Condition: Rainy",
        "chennai": "Temperature: 32°C, Humidity: 75%, Condition: Partly Cloudy",
    }

    return weather_info + weather_data.get(city.lower(), "Weather data not available for this city.")


@tool
def add_numbers(a: float, b: float) -> str:
    """Add two numbers and return the result.
    Args:
        a: First number.
        b: Second number.
    Returns:
        Sum of the two numbers.
    """
    result = a + b
    return f"The sum of {a} and {b} is {result}."

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

tools=[get_weather, add_numbers]

llm_with_tools=llm.bind_tools(tools)
question = input("Ask a question: ")
messages = [("user", question)]
while True:
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    if not response.tool_calls:
        print("Final Response:", response.content)
        break
    for tool_call in response.tool_calls:
        print("Selected tool:", tool_call["name"])
        print("Tool arguments:", tool_call["args"])
        if tool_call["name"] == "get_weather":
            result = get_weather.invoke(tool_call["args"])
        elif tool_call["name"] == "add_numbers":
            result = add_numbers.invoke(tool_call["args"])
        else:
            result = "Unknown tool."
        print("Tool result:", result)
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )
