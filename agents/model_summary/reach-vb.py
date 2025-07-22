import asyncio
from mcp_agent.core.fastagent import FastAgent

fast = FastAgent("auto-vb", ignore_unknown_args=True)

@fast.agent(
    name="reach_vb",
    servers=["prompts"],
    instruction="""You are `vb` - AKA `@reach_vb` an insightful commentator on
    the AI industry. You produce engaging summaries for x dot com, the everything app."""
)

async def main():
    async with fast.run() as agent:
        await agent.reach_vb.apply_prompt("auto-vb")
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
