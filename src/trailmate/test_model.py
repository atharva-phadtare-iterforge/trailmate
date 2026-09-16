import asyncio

from trailmate.agent.model import TrailMateModel
from trailmate.agent.tools.registry import get_tool_schemas


async def main():
    model = TrailMateModel()

    response = await model.complete(
        messages=[
            {
                "role": "user",
                "content": "Find me an easy trail with a waterfall.",
            }
        ],
        tools=get_tool_schemas(),
    )

    print("CONTENT:")
    print(response.content)

    print("\nTOOL CALLS:")
    print(response.tool_calls)


if __name__ == "__main__":
    asyncio.run(main())