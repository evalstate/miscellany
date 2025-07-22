import asyncio
import sys
from typing import List
from mcp import GetPromptResult
from mcp_agent import PromptMessageMultipart
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.fastagent import FastAgent
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
    instruction="You are a helpful AI Agent",
    use_history=True,
    servers=["model_server", "prompts"],
)
@fast.agent(
    name="url_picker",
    instruction="""
We are researching a Machine Learning model. You will be presented with a set of URLs
extracted from the Model Card. Assess which URLs are likely to contain useful additional
content, and indicate whether they should be fetched with a reason for your decision. 

Change  arXiv 'PDF' urls to be 'HTML' instead. for example: 

https://arxiv.org/pdf/2507.14311v1 becomes https://arxiv.org/html/2507.14311v1 

Return the adjusted URL in the supplied `url` field.
""",
)
@fast.agent(
    name="research_fetch",
    use_history=False,
    servers=["pulse_fetch", "model_server", "prompts"],
    instruction="""
You are an AI Agent that fetches information from the internet to assist in research.
Your output will later be used to generate a summary, so optimise your responses for clarity and relevance.
""",
)

async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:

        await agent.reach_vb.apply_prompt("auto-vb")

        model_id = None
        if "--research" in sys.argv:
            model_id = sys.argv[sys.argv.index("--research") + 1]

        if not model_id:
            print(
                f"WARNING: Use --research <model_id> to specify a model to research. Using default ${DEFAULT_RESEARCH_MODEL}"
            )
            model_id = DEFAULT_RESEARCH_MODEL

        model: GetPromptResult = await agent.summariser.get_prompt(
            "Model Card", {"model_id": model_id}
        )

        urls: str = extract_urls_from_model_card(model.messages[-1].content.text)
        to_access, _ = await agent.url_picker.structured(
            [Prompt.user(urls)], ExtraInformation
        )
        sources: List[tuple] = []
        if to_access:
            await agent.research_fetch.apply_prompt(
                "Model Card Template", {"model_id": model_id}, as_template=True
            )
            for URLWithReason in to_access.urls:
                if URLWithReason.fetch:
                    fetch_prompt = await agent.research_fetch.get_prompt(
                        "fetch-prompt", {"url": URLWithReason.url}
                    )
                    messages = await agent.research_fetch.generate(
                        PromptMessageMultipart.from_get_prompt_result(fetch_prompt)
                    )
                    content = messages.last_text()
                    if "NO USEFUL CONTENT" not in content:
                        sources.append((URLWithReason.url, content))

        research_report = format_research_report(sources)

        # Send the combined information to the user
        await agent.summariser.apply_prompt(
            "Model Card Template", {"model_id": model_id}
        )
        final_report = await agent.summariser.apply_prompt(
            "summary-prompt", {"research_report": research_report}
        )

        await agent.reach_vb.send(final_report)
        # show the interactive prompt for follow up questions
        await agent.interactive("summariser")


if __name__ == "__main__":
    asyncio.run(main())
