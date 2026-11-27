# Agents Practice

Small Python examples for exploring Google Gemini with LangChain. The project demonstrates a FastAPI chat endpoint and several ways to connect an LLM to tools, including weather lookups and Wikipedia summaries.

## Contents

| File | Purpose |
| --- | --- |
| `app.py` | FastAPI `POST /chat` endpoint that forwards a question to Gemini. |
| `main.py` | Minimal standalone Gemini prompt example. |
| `single_tool_pattern.py` | Lets Gemini call a weather tool, then returns a final answer. |
| `multiple_tool_calling.py` | Offers weather, city, and country information tools; handles multiple tool calls in one response. |
| `sequential_tools.py` | Extracts a destination from a prompt, calls city and weather tools in sequence, and asks Gemini to compose a response. |
| `reAct_tool.py` | ReAct-style loop where Gemini can call city and weather tools for up to five rounds. |
| `test.py` | Small `hello` print example. |

## Requirements

- Python 3.10 or newer
- A Google Gemini API key
- Internet access for Gemini and the weather/Wikipedia tools

## Setup

Run these commands from the `agents_practice` directory:

```bash
python -m venv venv
```

Activate the environment:

```bash
# Windows PowerShell
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
pip install requests
```

The tool examples use `requests` to access Open-Meteo and Wikipedia. It is installed separately because it is not currently listed in `requirements.txt`.

Create a `.env` file in this directory with your Gemini key:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Keep the key private; do not commit the `.env` file.

## Run the API

Start the development server from `agents_practice`:

```bash
uvicorn app:app --reload
```

Send a request to `http://127.0.0.1:8000/chat`:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Python?"}'
```

The response contains the submitted question and Gemini's response. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Run an example

The tool scripts prompt for a question in the terminal. For example:

```bash
python single_tool_pattern.py
python multiple_tool_calling.py
python sequential_tools.py
python reAct_tool.py
```

The standalone prompt example is `python main.py`. It currently needs `import os` added to `main.py` before it can run, because it calls `os.environ.get(...)` without importing `os`.

## Notes

- The API and tool examples use Gemini model names configured in their source files; availability depends on your Google AI account.
- The tool scripts make live requests to Open-Meteo and Wikipedia. Weather values are current lookup results, not forecasts tailored by this project.
- `requirements.txt` lists the main LangChain, Gemini, dotenv, FastAPI, and Uvicorn packages.
