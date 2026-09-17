from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

langfuse = get_client()

with langfuse.start_as_current_observation(
    as_type="span",
    name="trailmate-test",
    input={"message": "Hello from TrailMate"},
) as span:

    span.update(
        output={
            "status": "Langfuse connection works"
        }
    )

langfuse.flush()

print("Langfuse test completed.")