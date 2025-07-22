import re
from typing import List
from pydantic import BaseModel


class URLWithReason(BaseModel):
    url: str
    fetch: bool
    reason: str


class ExtraInformation(BaseModel):
    urls: List[URLWithReason]


def format_research_report(sources: List[tuple]) -> str:
    if not sources:
        return "\n<model_research />\n"
    
    report = "\n<model_research>\n"
    for url, content in sources:
        report += f'   <research source="{url}">\n'
        report += f"{content}\n"
        report += "   </research>\n"
    report += "\n</model_research>\n"
    
    return report


def extract_urls_from_model_card(model_card_content: str) -> str:
    try:
        # Extract URLs using regex
        url_pattern = re.compile(r"https?://[^\s\)\]\>\'\"`]+")
        urls = url_pattern.findall(model_card_content)

        if urls:
            # Preserve order while removing duplicates
            unique_urls = list(dict.fromkeys(urls))
            return "\n".join(unique_urls)
        else:
            return ""
    except Exception as e:
        raise Exception(f"Failed to extract URLs from model card: {e}")
