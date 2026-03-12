import logging
import os
from collections.abc import AsyncIterable
from enum import Enum
from typing import Annotated, Any, Literal

import openai
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from agent_framework import Agent, AgentSession, BaseChatClient, tool
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.openai import OpenAIChatClient

logger = logging.getLogger(__name__)
load_dotenv()


class ChatServices(str, Enum):
    """Supported chat completion services."""

    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"


service_id = "default"


def get_chat_completion_service(service_name: ChatServices) -> BaseChatClient:
    """Return a configured chat completion service."""
    if service_name == ChatServices.AZURE_OPENAI:
        return _get_azure_openai_chat_completion_service()
    if service_name == ChatServices.OPENAI:
        return _get_openai_chat_completion_service()
    raise ValueError(f"Unsupported service name: {service_name}")


def _get_azure_openai_chat_completion_service() -> AzureOpenAIChatClient:
    """Return Azure OpenAI chat service with MI or API key auth."""
    endpoint = os.getenv("gpt_endpoint")
    deployment_name = os.getenv("gpt_deployment")
    api_version = os.getenv("gpt_api_version")
    api_key = os.getenv("gpt_api_key")

    if not endpoint:
        raise ValueError("gpt_endpoint is required")
    if not deployment_name:
        raise ValueError("gpt_deployment is required")
    if not api_version:
        raise ValueError("gpt_api_version is required")

    if not api_key:
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

        async_client = openai.AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )

        return AzureOpenAIChatClient(
            service_id=service_id,
            deployment_name=deployment_name,
            async_client=async_client,
        )

    return AzureOpenAIChatClient(
        service_id=service_id,
        deployment_name=deployment_name,
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def _get_openai_chat_completion_service() -> OpenAIChatClient:
    """Return OpenAI chat completion service."""
    return OpenAIChatClient(
        service_id=service_id,
        model_id=os.getenv("OPENAI_MODEL_ID"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


@tool(
    name="get_products",
    description="Retrieves a set of products based on a natural language user query.",
)
def get_products(
    question: Annotated[
        str,
        "Natural language query to retrieve products, e.g. 'What kinds of paint rollers do you have in stock?'",
    ],
) -> list[dict[str, Any]]:
    """Return a mock product catalog for workshop usage."""
    try:
        _ = question  # Query filtering is intentionally mocked in this workshop step.
        return [
            {
                "id": "1",
                "name": "Eco-Friendly Paint Roller",
                "type": "Paint Roller",
                "description": "A high-quality, eco-friendly paint roller for smooth finishes.",
                "punchLine": "Roll with the best, paint with the rest!",
                "price": 15.99,
            },
            {
                "id": "2",
                "name": "Premium Paint Brush Set",
                "type": "Paint Brush",
                "description": "A set of premium paint brushes for detailed work and fine finishes.",
                "punchLine": "Brush up your skills with our premium set!",
                "price": 25.49,
            },
            {
                "id": "3",
                "name": "All-Purpose Paint Tray",
                "type": "Paint Tray",
                "description": "A durable paint tray suitable for all types of rollers and brushes.",
                "punchLine": "Tray it, paint it, love it!",
                "price": 9.99,
            },
        ]
    except Exception:
        logger.exception("Product recommendation failed.")
        return []


class ResponseFormat(BaseModel):
    """Structured response shape from the product management agent."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class AgentFrameworkProductManagementAgent:
    """Wrap Agent Framework for product-management chat interactions."""

    agent: Agent
    session: AgentSession | None = None

    def __init__(self) -> None:
        chat_service = get_chat_completion_service(ChatServices.AZURE_OPENAI)

        marketing_agent = Agent(
            client=chat_service,
            name="MarketingAgent",
            instructions=(
                "You specialize in planning and recommending marketing strategies for products. "
                "This includes identifying target audiences, making product descriptions better, "
                "and suggesting promotional tactics. Your goal is to help businesses effectively "
                "market their products and reach their desired customers."
            ),
        )

        ranker_agent = Agent(
            client=chat_service,
            name="RankerAgent",
            instructions=(
                "You specialize in ranking and recommending products based on various criteria. "
                "This includes analyzing product features, customer reviews, and market trends to "
                "provide tailored suggestions. Your goal is to help customers find the best products "
                "for their needs."
            ),
        )

        product_agent = Agent(
            client=chat_service,
            name="ProductAgent",
            instructions=(
                "You specialize in handling product-related requests from customers and employees. "
                "This includes providing a list of products, identifying available quantities, "
                "providing product prices, and giving product descriptions as they exist in the product catalog. "
                "Your goal is to assist customers promptly and accurately with all product-related inquiries. "
                "You are a helpful assistant that MUST use the get_products tool to answer all the questions from user. "
                "You MUST NEVER answer from your own knowledge UNDER ANY CIRCUMSTANCES. "
                "You MUST only use products from the get_products tool to answer product-related questions. "
                "Do not ask the user for more information about the products; instead use the get_products tool to find the "
                "relevant products and provide the information based on that. "
                "Do not make up any product information. Use only the product information from the get_products tool."
            ),
            tools=get_products,
        )

        self.agent = Agent(
            client=chat_service,
            name="ProductManagerAgent",
            instructions=(
                "Your role is to carefully analyze the user's request and respond as best as you can. "
                "Your primary goal is precise and efficient delegation to ensure customers and employees receive "
                "accurate and specialized assistance promptly. "
                "For requests that ask to improve messaging, product descriptions, positioning, promotions, or persuasive copy, "
                "you MUST delegate to the MarketingAgent first. "
                "For requests that ask to compare, rank, prioritize, or recommend one best option under constraints, "
                "you MUST delegate to the RankerAgent first. "
                "For factual catalog questions such as product list, price, and product details, "
                "you MUST delegate to the ProductAgent. "
                "You may use these agents in conjunction with each other to provide comprehensive responses to user queries."
            ),
            tools=[
                product_agent.as_tool(),
                marketing_agent.as_tool(),
                ranker_agent.as_tool(),
            ],
        )

    async def invoke(self, user_input: str, session_id: str) -> dict[str, Any]:
        """Handle non-streaming tasks."""
        await self._ensure_session_exists(session_id)

        response = await self.agent.run(
            messages=user_input,
            session=self.session,
            response_format=ResponseFormat,
        )
        raw_text = getattr(response, "text", None)
        if not raw_text:
            raw_text = str(response)
        return self._get_agent_response(raw_text)

    async def stream(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """Handle streaming tasks."""
        await self._ensure_session_exists(session_id)

        chunks: list[str] = []
        async for chunk in self.agent.run_stream(messages=user_input, session=self.session):
            text = getattr(chunk, "text", None)
            if text:
                chunks.append(str(text))

        if chunks:
            yield self._get_agent_response("".join(chunks))

    def _get_agent_response(self, message: str) -> dict[str, Any]:
        """Extract and normalize structured response content."""
        default_response = {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }

        try:
            structured_response = ResponseFormat.model_validate_json(message)
        except ValidationError:
            logger.info("Message did not come in JSON format.")
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": message,
            }
        except Exception:
            logger.exception("An unexpected error occurred while processing the message.")
            return default_response

        response_map = {
            "input_required": {"is_task_complete": False, "require_user_input": True},
            "error": {"is_task_complete": False, "require_user_input": True},
            "completed": {"is_task_complete": True, "require_user_input": False},
        }

        mapped = response_map.get(structured_response.status)
        if mapped:
            return {**mapped, "content": structured_response.message}

        return default_response

    async def _ensure_session_exists(self, session_id: str) -> None:
        """Create/reuse Agent Framework session for the current conversation."""
        if self.session is None or self.session.service_session_id != session_id:
            self.session = self.agent.create_session(session_id=session_id)
