import os
import base64
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import numpy as np
import time
from typing import Any, cast

# Load environment variables (Azure endpoint, deployment, keys, etc.)
load_dotenv()

# Retrieve credentials from .env file or environment
endpoint = os.getenv("gpt_endpoint")
deployment = os.getenv("gpt_deployment")
api_key = os.getenv("gpt_api_key")
api_version = os.getenv("gpt_api_version")

if not endpoint or not deployment or not api_key or not api_version:
    raise ValueError(
        "Missing required GPT environment variables: gpt_endpoint, gpt_deployment, gpt_api_key, gpt_api_version"
    )

gpt_endpoint: str = endpoint
gpt_deployment: str = deployment
gpt_api_key: str = api_key
gpt_api_version: str = api_version

# Initialize Azure OpenAI client for GPT model
client = AzureOpenAI(
    azure_endpoint=gpt_endpoint,
    api_key=gpt_api_key,
    api_version=gpt_api_version,
)


def generate_response(text_input):
    start_time = time.time()
    """
    Input:
        text_input (str): The user's chat input.

    Output:
        response (str): A Markdown-formatted response from the agent.
    """

    # Prepare the full chat prompt with system and user messages
    chat_prompt: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": """You are a helpful assistant working for Zava, a company that specializes in offering products to assist homeowners with do-it-yourself projects.
                Respond to customer inquiries with relevant product recommendations and DIY tips. If a customer asks for paint, suggest one of the following three colors: blue, green, and white.
                If a customer asks for something not related to a DIY project, politely inform them that you can only assist with DIY-related inquiries.
                Zava has a variety of store locations across the country. If a customer asks about store availability, direct the customer to the Miami store.
            """,
        },
        {"role": "user", "content": text_input},
    ]

    # Call Azure OpenAI chat API
    completion = client.chat.completions.create(
        model=cast(Any, gpt_deployment),
        messages=cast(Any, chat_prompt),
        max_completion_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
    )
    end_sum = time.time()
    print(f"generate_response Execution Time: {end_sum - start_time} seconds")

    # Return response content. GPT responses can be a plain string or structured blocks.
    if not completion.choices:
        return "No response was returned by the model."

    message = completion.choices[0].message
    content = message.content

    if isinstance(content, str) and content.strip():
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text")
                content_value = part.get("content")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
                elif isinstance(content_value, str):
                    text_parts.append(content_value)
        combined = "\n".join([t for t in text_parts if t]).strip()
        if combined:
            return combined

    # Last-resort fallback for debugging unexpected payloads
    return json.dumps(completion.model_dump(), ensure_ascii=False)
