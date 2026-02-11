import asyncio
from pathlib import Path
import sys
from typing import List
from fast_agent import FastAgent, Prompt
from mcp.types import GetPromptResult, PromptMessage, TextContent
from research_utils import (
    ExtraInformation,
    format_research_report,
    extract_urls_from_model_card,
)

fast = FastAgent("Model Summary", ignore_unknown_args=True)

DEFAULT_RESEARCH_MODEL = "HuggingFaceTB/SmolLM3-3B"

# black-forest-labs/FLUX.1-Kontext-dev
# Qwen/Qwen3-235B-A22B-GGUF
#"moonshotai/Kimi-K2-Instruct


# Define the agent
@fast.agent(
    name="summariser",
    instruction="""
You are a helpful AI Agent. You will be producing a summary report for a model. In addition, you must also produce:

a) A "Prompting Guide" that describes the best way to prompt the model.
b) An example "Tool Description" and "Parameter Descriptions" to help an LLM Produce optimal outputs for the model",
""",
    use_history=True,
    servers=["model_server", "prompts"],
)
@fast.agent(
    name="url_picker",
    instruction=Path("url_prompt.md")
)
@fast.agent(
    name="research_fetch",
    use_history=False,
    servers=["pulse_fetch", "model_server", "prompts"],
    instruction="""
You are an AI Agent that fetches information from the internet to assist in research.
Your output will later be used to produce a prompting guide, so optimise your responses for clarity and relevance.
""",
)

async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.interactive()
        model_id = None
        if "--research" in sys.argv:
            model_id = sys.argv[sys.argv.index("--research") + 1]

        if not model_id:
            print(
                f"WARNING: Use --research <model_id> to specify a model to research. Using default ${DEFAULT_RESEARCH_MODEL}"
            )
            model_id = DEFAULT_RESEARCH_MODEL

        model: GetPromptResult = await agent.summariser.get_prompt(
            "Model Details", {"model_id": model_id}
        )

        urls: str = extract_urls_from_model_card(model.messages[-1].content.text)
        to_access, _ = await agent.url_picker.structured(
            [Prompt.user(urls)], ExtraInformation
        )
        sources: List[tuple] = []
        if to_access:
            model.messages[-1].role="assistant"
            await agent.research_fetch.apply_prompt(model, as_template=True)
            for URLWithReason in to_access.urls:
                if URLWithReason.fetch:
                    fetch_prompt: GetPromptResult= await agent.research_fetch.get_prompt(
                        "fetch-prompt", {"url": URLWithReason.url}
                    )
                    messages = await agent.research_fetch.generate(fetch_prompt.messages)
                    content = messages.last_text() or ""
                    if "NO USEFUL CONTENT" not in content:
                        sources.append((URLWithReason.url, content))

        research_report = format_research_report(sources)

        model.messages.append(PromptMessage(role="assistant",content=TextContent(type="text",text="Thank you. I will refer to this model card in my future responses")))
        await agent.summariser.apply_prompt(model,as_template=True)
        
        await agent.summariser.apply_prompt(
            "summary-prompt", {"research_report": research_report}
        )

        await agent.interactive("summariser")


if __name__ == "__main__":
    asyncio.run(main())
