<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    variant?: "problem" | "tool" | "solution";
  }>(),
  { variant: "solution" },
);
</script>

<template>
  <section
    class="http-request-panel deck-panel"
    :class="`http-request-panel--${props.variant}`"
  >
    <template v-if="props.variant === 'problem'">
      <h2>Before 2026-07-28</h2>

      <div class="http-request-line">
        <strong>POST /mcp/ HTTP/1.1</strong>
        <span>Host: hf.co</span>
        <span>MCP-Protocol-Version: 2025-06-18</span>
        <span>Content-Type: application/json</span>
      </div>

      <div class="http-json http-json--packet">
        <div class="http-json-line">{</div>
        <div class="http-json-line http-json-line--indent"><em>"jsonrpc"</em>: <strong>"2.0"</strong>,</div>
        <div class="http-json-line http-json-line--indent"><em>"id"</em>: <mark><strong>1</strong></mark>,</div>
        <div class="http-json-line http-json-line--indent"><em>"method"</em>: <mark>"tools/call"</mark>,</div>
        <div class="http-json-line http-json-line--indent"><em>"params"</em>: {</div>
        <div class="http-json-line http-json-line--indent-2"><em>"name"</em>: <mark>"hf.run_in_sandbox"</mark>,</div>
        <div class="http-json-line http-json-line--indent-2"><em>"arguments"</em>: {</div>
        <div class="http-json-line http-json-line--indent-3"><em>"sandbox_id"</em>: <mark>"sbx-7f3c"</mark>,</div>
        <div class="http-json-line http-json-line--indent-3"><em>"command"</em>: <strong>"python train.py"</strong></div>
        <div class="http-json-line http-json-line--indent-2">}</div>
        <div class="http-json-line http-json-line--indent">}</div>
        <div class="http-json-line">}</div>
      </div>
    </template>

    <template v-else-if="props.variant === 'tool'">
      <div class="http-schema-layout">
        <div class="http-tool-panel http-tool-panel--schema">
          <span>tool definition</span>
          <div class="http-json-line">{</div>
          <div class="http-json-line http-json-line--indent"><em>"name"</em>: <mark>"hf.run_in_sandbox"</mark>,</div>
          <div class="http-json-line http-json-line--indent"><em>"inputSchema"</em>: {</div>
          <div class="http-json-line http-json-line--indent-2"><em>"properties"</em>: {</div>
          <div class="http-json-line http-json-line--indent-3"><em>"command"</em>: { <em>"type"</em>: <strong>"string"</strong> },</div>
          <div class="http-json-line http-json-line--indent-3"><em>"sandbox_id"</em>: {</div>
          <div class="http-json-line http-json-line--indent-4"><em>"type"</em>: <strong>"string"</strong>,</div>
          <div class="http-json-line http-json-line--indent-4"><mark>"x-mcp-header": "Sandbox-Id"</mark></div>
          <div class="http-json-line http-json-line--indent-3">}</div>
          <div class="http-json-line http-json-line--indent-2">}</div>
          <div class="http-json-line http-json-line--indent">}</div>
          <div class="http-json-line">}</div>
        </div>

        <div class="http-schema-notes">
          <div>
            <strong>Explicit application state</strong>
            <span><code>sbx-7f3c</code> already travels as ordinary tool input.</span>
          </div>
          <div>
            <strong>One declared routing key</strong>
            <span><code>x-mcp-header</code> mirrors it as <code>Mcp-Param-Sandbox-Id</code>.</span>
          </div>
          <div>
            <strong>Still one source of truth</strong>
            <span>The header must match the JSON body—or the request is rejected.</span>
          </div>
        </div>
      </div>
    </template>

    <template v-else>
      <h2>Router can see</h2>

      <div class="http-tool-panel">
        <span>standard request metadata</span>
        <div class="http-code-row">
          <em>method</em><mark>tools/call</mark>
        </div>
        <div class="http-code-row">
          <em>name</em><mark>hf.run_in_sandbox</mark>
        </div>
      </div>

      <div class="http-request-line http-request-line--solution">
        <strong>POST /mcp/ HTTP/1.1</strong>
        <span>Host: hf.co</span>
        <span>MCP-Protocol-Version: 2026-07-28</span>
        <span><mark>Mcp-Method: tools/call</mark></span>
        <span><mark>Mcp-Name: hf.run_in_sandbox</mark></span>
      </div>

      <div class="http-json http-json--packet">
        <div class="http-json-line">{</div>
        <div class="http-json-line http-json-line--indent"><em>"jsonrpc"</em>: <strong>"2.0"</strong>,</div>
        <div class="http-json-line http-json-line--indent"><em>"id"</em>: <strong>2</strong>,</div>
        <div class="http-json-line http-json-line--indent"><em>"method"</em>: <mark>"tools/call"</mark>,</div>
        <div class="http-json-line http-json-line--indent"><em>"params"</em>: {</div>
        <div class="http-json-line http-json-line--indent-2"><em>"_meta"</em>: {</div>
        <div class="http-json-line http-json-line--indent-3 http-json-line--meta"><em>"io.modelcontextprotocol/protocolVersion"</em>: <strong>"2026-07-28"</strong>,</div>
        <div class="http-json-line http-json-line--indent-3 http-json-line--meta"><em>"io.modelcontextprotocol/clientInfo"</em>: { <em>"name"</em>: <strong>"HF"</strong>, <em>"version"</em>: <strong>"1"</strong> },</div>
        <div class="http-json-line http-json-line--indent-3 http-json-line--meta"><em>"io.modelcontextprotocol/clientCapabilities"</em>: {}</div>
        <div class="http-json-line http-json-line--indent-2">},</div>
        <div class="http-json-line http-json-line--indent-2"><em>"name"</em>: <mark>"hf.run_in_sandbox"</mark>,</div>
        <div class="http-json-line http-json-line--indent-2"><em>"arguments"</em>: { <em>"sandbox_id"</em>: <strong>"sbx-7f3c"</strong>, <em>"command"</em>: <strong>"python train.py"</strong> }</div>
        <div class="http-json-line http-json-line--indent">}</div>
        <div class="http-json-line">}</div>
      </div>
    </template>
  </section>
</template>
