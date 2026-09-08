import re
from pathlib import Path

def chunk_markdown(file_path: Path) -> list[dict]:
    text = file_path.read_text(encoding="utf-8")

    lines = text.splitlines()

    trail_name = ""
    current_section = None
    current_content = []

    chunks = []

    for line in lines:
        # Main title: # Trail Name
        if line.startswith("# ") and not line.startswith("## "):
            trail_name = line[2:].strip()
            continue

        # Section heading: ## Overview, ## Details, etc.
        if line.startswith("## "):
            if current_section and current_content:
                chunks.append(
                    {
                        "trail_id": file_path.stem,
                        "trail_name": trail_name,
                        "section_name": current_section,
                        "chunk_text": build_chunk_text(
                            trail_name,
                            current_section,
                            current_content,
                        ),
                    }
                )

            current_section = line[3:].strip()
            current_content = []
            continue

        # Ignore content before the first ## heading
        if current_section:
            current_content.append(line)

    # Add final section
    if current_section and current_content:
        chunks.append(
            {
                "trail_id": file_path.stem,
                "trail_name": trail_name,
                "section_name": current_section,
                "chunk_text": build_chunk_text(
                    trail_name,
                    current_section,
                    current_content,
                ),
            }
        )

    return chunks


def build_chunk_text(
    trail_name: str,
    section_name: str,
    content: list[str],
) -> str:
    body = "\n".join(content).strip()

    body = re.sub(
        r"\n{3,}",
        "\n\n",
        body,
    )

    return (
        f"{trail_name} — {section_name}\n\n"
        f"{body}"
    )