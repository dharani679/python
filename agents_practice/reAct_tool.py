
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from multiple_tool_calling import get_city_info, get_weather



load_dotenv()
tools = [
    get_weather,
    get_city_info
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

messages = [
    SystemMessage(
        content="""
        You are a helpful travel assistant.

        Use the available tools whenever external city or weather
        information is needed.

        After receiving a tool result, decide whether another
        tool is necessary.

        Once you have enough information, provide a final answer.
        """
    ),
    HumanMessage(content=question)
]

max_iterations = 5

for step in range(max_iterations):
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    if not response.tool_calls:

        print("\nFinal Answer:")
        print(response.content)
        break
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        print(f"Selected Tool: {tool_name}")
        print(f"Arguments: {tool_args}")
        selected_tool = tools_by_name.get(tool_name)
        if selected_tool is None:
            tool_result = f"Unknown tool: {tool_name}"
        else:
            try:
                tool_result = selected_tool.invoke(tool_args)
            except Exception as e:
                tool_result = f"Tool execution failed: {e}"
        print("\nObservation:")
        print(tool_result)
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_id
            )
        )
else:
    print("Maximum iterations reached.")
    print("The agent could not complete the task.")
