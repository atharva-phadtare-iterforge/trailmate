import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from litellm import acompletion


class TrailMateModel:
    def __init__(
        self,
        model: str = "hiking-small",
        base_url: str = "http://localhost:4000/v1",
        api_key: str = "sk-1234",
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    # --------------------------------------------------
    # CONVERT LANGCHAIN MESSAGE -> OPENAI MESSAGE
    # --------------------------------------------------

    def convert_message(self, message) -> dict:

        if isinstance(message, HumanMessage):
            return {
                "role": "user",
                "content": str(message.content),
            }

        if isinstance(message, SystemMessage):
            return {
                "role": "system",
                "content": str(message.content),
            }

        if isinstance(message, AIMessage):

            result = {
                "role": "assistant",
                "content": message.content or None,
            }

            if message.tool_calls:

                result["tool_calls"] = []

                for tool_call in message.tool_calls:

                    result["tool_calls"].append(
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(
                                    tool_call["args"]
                                ),
                            },
                        }
                    )

            return result

        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": str(message.content),
            }

        raise TypeError(
            f"Unsupported message type: {type(message)}"
        )

    # --------------------------------------------------
    # STREAM
    # --------------------------------------------------

    async def stream(
        self,
        messages: list,
        tools: list[dict] | None = None,
    ):

        converted_messages = [
            self.convert_message(message)
            for message in messages
        ]

        kwargs = {
            "model": f"openai/{self.model}",
            "messages": converted_messages,
            "api_base": self.base_url,
            "api_key": self.api_key,
            "stream": True,
            "extra_body": {
                "think": False,
            },
        }

        # --------------------------------------------------
        # ADD TOOLS
        # --------------------------------------------------

        if tools:

            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # --------------------------------------------------
        # CALL LITELLM
        # --------------------------------------------------

        response = await acompletion(**kwargs)

        # --------------------------------------------------
        # STREAM RESPONSE
        # --------------------------------------------------

        async for chunk in response:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            content = getattr(
                delta,
                "content",
                None,
            )

            if content:
                yield content

    # --------------------------------------------------
    # COMPLETE
    # --------------------------------------------------

    async def complete(
        self,
        messages: list,
        tools: list[dict] | None = None,
    ) -> AIMessage:

        converted_messages = [
            self.convert_message(message)
            for message in messages
        ]

        kwargs = {
            "model": f"openai/{self.model}",
            "messages": converted_messages,
            "api_base": self.base_url,
            "api_key": self.api_key,
            "extra_body": {
                "think": False,
            },
        }

        # --------------------------------------------------
        # ADD TOOLS
        # --------------------------------------------------

        if tools:

            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # --------------------------------------------------
        # CALL LITELLM
        # --------------------------------------------------

        response = await acompletion(**kwargs)

        # --------------------------------------------------
        # EXTRACT MESSAGE
        # --------------------------------------------------

        message = response.choices[0].message

        # --------------------------------------------------
        # CONVERT TOOL CALLS
        # --------------------------------------------------

        tool_calls = []

        for tool_call in message.tool_calls or []:

            function = tool_call.function

            arguments = function.arguments

            # Some providers may already return a dictionary.
            if isinstance(arguments, str):

                try:
                    arguments = json.loads(arguments)

                except json.JSONDecodeError:
                    arguments = {}

            tool_calls.append(
                {
                    "name": function.name,
                    "args": arguments,
                    "id": tool_call.id,
                }
            )

        # --------------------------------------------------
        # RETURN LANGCHAIN MESSAGE
        # --------------------------------------------------

        return AIMessage(
            content=message.content or "",
            tool_calls=tool_calls,
        )