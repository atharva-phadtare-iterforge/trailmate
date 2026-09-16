import json

import requests

from langchain_core.tools import tool


SEARCH_SERVICE_URL = "http://127.0.0.1:8002/search"


@tool
def search_trails_tool(query: str) -> str:
    """
    # PURPOSE

    Search the TrailMate trail database for trails matching the user's
    requirements.

    # SEARCHABLE INFORMATION

    The search can use concrete trail requirements such as difficulty, distance,
    length or range, location or proximity, elevation gain, dog access, specific
    trail characteristics or Best For requirements, and specific trail names or
    trail locations.

    Multiple requirements can be combined in a single search.

    # SEARCH BEHAVIOR

    Search the TrailMate database using the requirements provided in the query.
    Preserve the user's relevant requirements when performing the search so
    that the returned results reflect the user's requested criteria.

    # SOURCE OF TRUTH

    The TrailMate database is the source of truth for trail information.
    Results must be based only on information available in the TrailMate
    database.

    Do not invent trails or trail-specific information. Do not supplement
    missing database information with general knowledge.

    # SPECIFIC TRAILS

    If the query identifies a specific trail or trail location, search the
    database for that trail first.

    If the requested trail is not present in the database, indicate that it was
    not found rather than providing information from outside the database.

    # NO MATCHES

    If no trails match the provided requirements, indicate that no matching
    trails were found in the TrailMate database.

    # INPUT

    The query contains the user's trail search requirements that should be used
    for the database search.

    Args:
        query: The user's trail search requirements.

    Returns:
        Trail information returned from the TrailMate database.
    """

    response = requests.post(
        SEARCH_SERVICE_URL,
        json={
            "query": query,
            "top_k": 5,
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])

    if not results:
        return "No matching trails were found in the TrailMate database."

    return json.dumps(results)
