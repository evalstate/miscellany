#!/usr/bin/env python3
"""
Server to get Model Card information
"""

import logging
from huggingface_hub import ModelCard

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import UserMessage, AssistantMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
mcp = FastMCP(name="model_server", instructions="Return a model card for a given model ID")


@mcp.prompt(name="Model Card", description="Enter the Model ID to get the model card")
def model_card(model_id: str) -> str:
    """Get model card information from Hugging Face Hub"""
    card = ModelCard.load(model_id)
    # Allow exception to propagate
    result = f"""
```
{str(card)}
```

"""
    return result


@mcp.prompt(
    name="Model Card Template", description="For use as an LLM Responder template"
)
def model_card_template(model_id: str) -> list[UserMessage | AssistantMessage]:
    card = ModelCard.load(model_id)
    # Allow exception to propagate
    return [
        UserMessage(f"""
Here is the model card for {model_id}:

```
{str(card)}
```

"""),
        AssistantMessage(
            "Thank you. I will refer to this model card in my future responses."
        ),
    ]


if __name__ == "__main__":
    # Run the server using stdio transport
    mcp.run(transport="stdio")
