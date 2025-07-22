# Model Summariser

## About

Simple workflow to produce a summary of a Model from the Hugging Face
Hub.

Modify `summary-prompt.md` to adjust the final summarisation prompt.

## Usage

Install [`uv`](https://docs.astral.sh/uv/) for running the Python code, and [`node.js`](https://nodejs.org/en) for the required MCP Servers. 

Create a venv and install dependencies:

```bash
# set up and use a virual python environment
uv venv
source .venv/bin/activate

# install dependencies
uv sync
```

By default it's configured to use OpenAI GPT-4.1 (that's what it's been tested with) either
set an API key (`OPENAI_API_KEY`) or edit the config file (`fast-agent setup` to create
a sample. Use `fast-agent check` to confirm API key is found.)

To run:

```bash
uv run summary.py --research <model_id>
```

If no research model id is specified, a sample default is used (`HuggingFaceTB/SmolLM3-3B`)

To save the results type `***SAVE_HISTORY <filename.md>`. To save the results as JSON use `***SAVE_HISTORY <filename.json>`
Use `CTRL+Y` to copy the plain markdown to the clipboard.

## Next Steps

### Tune Summarisation Prompt

- Load the last `---USER` turn from the saved history and re-run/tune the summarisation prompt manually, or with an
_eval/optimiser_ and ask for an optimised prompt.

- Add date awareness to the prompt to contextualise report summaries.

- The `summary-prompt.md` can also use in-context learning with examples between  `---USER` and `---ASSISTANT` delimeters.


### Improve Fetching

For markdown conversion it's possible to use `jiniai/readerlm2` locally for content extraction (add new tool to MCP
server to attempt CURL extraction).

Add additional "fetch" mechanisms as a fallback for 400's or errors (e.g. add `mcp-server-fetch` as an additional option), or
even add a "playwright" agent if wanted.
