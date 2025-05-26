# Gemini Code: OpenRouter Bridge for Anthropic API Compatibility

This server acts as a bridge, allowing you to use Anthropic-compatible clients (like Claude Code) with multiple LLM providers through OpenRouter. It translates API requests and responses between the Anthropic format and various model providers via LiteLLM and OpenRouter.

![Anthropic API Proxy](image.png)

## Features

- **Anthropic API Compatibility**: Use clients designed for Anthropic's API with any OpenRouter-supported model.
- **Multi-Provider Model Access**: Access models from Anthropic, OpenAI, Google, Meta, Mistral, Cohere, and more through OpenRouter.
- **Model Mapping**: Maps Anthropic model aliases (e.g., `haiku`, `sonnet`) to specific OpenRouter models.
- **LiteLLM Integration**: Leverages LiteLLM for robust interaction with multiple model providers.
- **Streaming Support**: Handles streaming responses for interactive experiences.
- **Tool Use Support**: Translates tool/function calling between formats.
- **Token Counting**: Provides a `/v1/messages/count_tokens` endpoint.

## Prerequisites

- An OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai/keys)).
- Python 3.8+.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/coffeegrind123/gemini-code.git # Or your fork
    cd gemini-code
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your OpenRouter API key:
    ```dotenv
    OPENROUTER_API_KEY="your-openrouter-api-key" # Required

    # Optional: Customize which OpenRouter models are used for Anthropic aliases
    # BIG_MODEL="anthropic/claude-3.5-sonnet"    # For 'sonnet' requests
    # SMALL_MODEL="anthropic/claude-3-haiku"     # For 'haiku' requests
    ```
    The server defaults to `anthropic/claude-3.5-sonnet` for `BIG_MODEL` (Claude's `sonnet`) and `anthropic/claude-3-haiku` for `SMALL_MODEL` (Claude's `haiku`).

5.  **Run the server**:

    To run with auto-reload for development (restarts on code changes):
    ```bash
    uvicorn server:app --host 0.0.0.0 --port 8083 --reload
    ```

    To run normally:
    ```bash
    python server.py
    ```
    (The `server.py` file has a `__main__` block that runs `uvicorn` for you.)

## Usage with Claude Code Client

1.  Ensure Claude Code CLI is installed (`npm install -g @anthropic-ai/claude-code`).
2.  Point the Claude Code client to your running proxy server:
    ```bash
    ANTHROPIC_BASE_URL=http://localhost:8083 claude
    ```
3.  For optimal performance, disable conversation history compacting after starting Claude Code:
    ```
    /config
    ```
    Select the option to disable history compacting when prompted. This ensures proper handling of context between the client and proxy.

4.  **Utilizing the CLAUDE.md file**:
    - The repository includes a `CLAUDE.md` file to improve model performance with Claude Code tooling.
    - Copy this file to any project directory where you'll be using Claude Code:
      ```bash
      cp /path/to/gemini-code/CLAUDE.md /your/project/directory/
      ```
    - When starting a new conversation with Claude Code in that directory, begin with:
      ```
      First read and process CLAUDE.md with intent. After understanding and agreeing to use the policies and practices outlined in the document, respond with YES
      ```
    - This ensures the model receives important context and instructions for better assistance.

## How It Works

1.  The proxy receives an API request formatted for Anthropic's Messages API.
2.  It validates and converts this request into a format LiteLLM can use with OpenRouter.
3.  The request is sent to the specified model via OpenRouter using LiteLLM.
4.  The model response is received (either streaming or complete).
5.  This response is converted back into the Anthropic Messages API format.
6.  The Anthropic-formatted response is sent back to the client.

## Model Mapping

- Requests for `claude-3-haiku...` or similar short model names containing "haiku" are mapped to the `SMALL_MODEL` (default: `anthropic/claude-3-haiku`).
- Requests for `claude-3-sonnet...` or similar short model names containing "sonnet" are mapped to the `BIG_MODEL` (default: `anthropic/claude-3.5-sonnet`).
- You can also specify full OpenRouter model names directly in your client requests (e.g., `openai/gpt-4o`, `google/gemini-pro-1.5`).

The server maintains a list of supported OpenRouter models and validates requests against this list.

## Supported Models

The proxy supports a wide range of models through OpenRouter, including:

- **Anthropic**: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-haiku`, `anthropic/claude-3-opus`
- **OpenAI**: `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-4-turbo`, `openai/gpt-3.5-turbo`
- **Google**: `google/gemini-pro-1.5`, `google/gemini-flash-1.5`
- **Meta**: `meta-llama/llama-3.1-405b-instruct`, `meta-llama/llama-3.1-70b-instruct`
- **Mistral**: `mistralai/mixtral-8x7b-instruct`
- **Cohere**: `cohere/command-r-plus`

## Endpoints

- `POST /v1/messages`: Main endpoint for sending messages to models (Anthropic compatible).
- `POST /v1/messages/count_tokens`: Estimates token count for a given set of messages (Anthropic compatible).
- `GET /`: Root endpoint, returns a welcome message.

## Logging

The server provides detailed logging, including colorized output for model mappings and request processing, to help with debugging and monitoring.

## Additional Files

### CLAUDE.md

The repository includes a special `CLAUDE.md` file that contains instructions to optimize model behavior when used with Claude Code tooling. This file is designed to:

- Help models better understand how to respond to Claude Code commands
- Improve code generation and project understanding
- Enhance tool-using capabilities

The file needs to be present in any directory where you run Claude Code, as Claude Code automatically reads from a `CLAUDE.md` file in the current working directory.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Thanks

Special thanks to https://github.com/1rgs/claude-code-proxy for inspiration and original code.
