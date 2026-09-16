import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from trailmate.agent.agent import trailmate_agent


async def run_turn(user_message: str, thread_id: str):

    result = await trailmate_agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=user_message)
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )

    messages = result["messages"]

    # --------------------------------------------------
    # USER INPUT
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("USER INPUT")
    print("=" * 60)
    print(user_message)

    # --------------------------------------------------
    # COLLECT EXECUTION INFORMATION
    # --------------------------------------------------

    decisions = []
    tools_called = []
    tool_inputs = []
    tool_results = []
    final_response = None

    for message in messages:

        # Agent decision
        if isinstance(message, AIMessage):

            if message.tool_calls:

                for tool_call in message.tool_calls:

                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    tools_called.append(tool_name)
                    tool_inputs.append(tool_args)

                    if tool_name == "search_trails_tool":
                        decisions.append(
                            "Search the trail database because "
                            "the user provided searchable trail requirements."
                        )

            elif message.content:
                final_response = message.content

        # Tool result
        elif isinstance(message, ToolMessage):

            tool_results.append(message.content)

    # --------------------------------------------------
    # AGENT DECISION
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("AGENT DECISION")
    print("=" * 60)

    if decisions:
        for decision in decisions:
            print(decision)
    else:
        print("Respond directly without using a tool.")

    # --------------------------------------------------
    # TOOL
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TOOL")
    print("=" * 60)

    if tools_called:
        for tool in tools_called:
            print(tool)
    else:
        print("No tool called")

    # --------------------------------------------------
    # TOOL INPUT
    # --------------------------------------------------

    if tool_inputs:

        print("\n" + "=" * 60)
        print("TOOL INPUT")
        print("=" * 60)

        for tool_input in tool_inputs:

            if "query" in tool_input:
                print(tool_input["query"])
            else:
                print(tool_input)

    # --------------------------------------------------
    # TOOL RESULT
    # --------------------------------------------------

    if tool_results:

        print("\n" + "=" * 60)
        print("TOOL RESULT")
        print("=" * 60)

        for tool_result in tool_results:

            # Make search results easier to read
            if "Trail:" in tool_result:

                trail_names = []

                for line in tool_result.splitlines():

                    if line.startswith("Trail:"):

                        trail_names.append(
                            line.replace("Trail:", "").strip()
                        )

                if trail_names:

                    print(
                        f"{len(trail_names)} matching trails found."
                    )

                    for index, trail_name in enumerate(
                        trail_names,
                        start=1,
                    ):
                        print(
                            f"{index}. {trail_name}"
                        )

                else:
                    print(tool_result)

            else:
                print(tool_result)

    # --------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RESPONSE")
    print("=" * 60)

    if final_response:
        print(final_response)
    else:
        print("No final response generated.")

    # --------------------------------------------------
    # GRAPH FLOW
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("GRAPH FLOW")
    print("=" * 60)

    print("User")

    for tool in tools_called:

        print("  ↓")
        print("Agent")

        print("  ↓")
        print(tool)

        if tool == "search_trails_tool":

            print("  ↓")
            print("Trail Database")

            print("  ↓")
            print("Search Results")

    print("  ↓")
    print("Agent")

    print("  ↓")
    print("Final Response")

    return result


async def main():

    thread_id = "day3-search-test"

    await run_turn(
        "Something easy with waterfalls.",
        thread_id,
    )


if __name__ == "__main__":
    asyncio.run(main())