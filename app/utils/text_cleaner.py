import re

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").strip()
    return text
