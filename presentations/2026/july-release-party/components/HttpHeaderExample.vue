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
      <h2>MCP over HTTP today</h2>

      <div class="http-request-line">
        <strong>POST /mcp/ HTTP/1.1</strong>
        <span>Host: hf.co/mcp</span>
        <span>Content-Type: application/json</span>
      </div>

      <div class="http-json http-json--packet">
        <div class="http-json-line">{</div>
        <div class="http-json-line http-json-line--indent"><em>"jsonrpc"</em>: <strong>"2.0"</strong>,</div>
        <div class="http-json-line http-json-line--indent"><em>"method"</em>: <mark>"tools/call"</mark>,</div>
        <div class="http-json-line http-json-line--indent"><em>"params"</em>: {</div>
        <div class="http-json-line http-json-line--indent-2"><em>"name"</em>: <mark>"hf.hub_search"</mark>,</div>
        <div class="http-json-line http-json-line--indent-2"><em>"arguments"</em>: {</div>
        <div class="http-json-line http-json-line--indent-3"><em>"query"</em>: <strong>"image generation models"</strong>,</div>
        <div class="http-json-line http-json-line--indent-3"><em>"type"</em>: <strong>"model"</strong></div>
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
          <div class="http-json-line http-json-line--indent"><em>"name"</em>: <mark>"hf.generate_image"</mark>,</div>
          <div class="http-json-line http-json-line--indent"><em>"inputSchema"</em>: {</div>
          <div class="http-json-line http-json-line--indent-2"><em>"properties"</em>: {</div>
          <div class="http-json-line http-json-line--indent-3"><em>"prompt"</em>: { <em>"type"</em>: <strong>"string"</strong> },</div>
          <div class="http-json-line http-json-line--indent-3"><em>"size"</em>: { <em>"type"</em>: <strong>"string"</strong> },</div>
          <div class="http-json-line http-json-line--indent-3"><em>"pipeline"</em>: {</div>
          <div class="http-json-line http-json-line--indent-4"><em>"type"</em>: <strong>"string"</strong>,</div>
          <div class="http-json-line http-json-line--indent-4"><mark>"x-mcp-header": "Pipeline"</mark></div>
          <div class="http-json-line http-json-line--indent-3">}</div>
          <div class="http-json-line http-json-line--indent-2">}</div>
          <div class="http-json-line http-json-line--indent">}</div>
          <div class="http-json-line">}</div>
        </div>

        <div class="http-schema-notes">
          <div>
            <strong>Ordinary tool input</strong>
            <span><code>prompt</code> and <code>size</code> stay in the JSON body.</span>
          </div>
          <div>
            <strong>One routable parameter</strong>
            <span><code>x-mcp-header</code> tells clients what to mirror into HTTP.</span>
          </div>
          <div>
            <strong>Infrastructure hint</strong>
            <span>The edge can choose image infrastructure without parsing the packet.</span>
          </div>
        </div>
      </div>
    </template>

    <template v-else>
      <h2>Router can see</h2>

      <div class="http-tool-panel">
        <span>tool definition</span>
        <div class="http-code-row">
          <em>name</em><mark>hf.generate_image</mark>
        </div>
        <div class="http-code-row">
          <em>x-mcp-header</em><mark>Pipeline</mark>
        </div>
      </div>

      <div class="http-request-line http-request-line--solution">
        <strong>POST /mcp/ HTTP/1.1</strong>
        <span>Host: hf.co/mcp</span>
        <span><mark>Mcp-Method: tools/call</mark></span>
        <span><mark>Mcp-Name: hf.generate_image</mark></span>
        <span><mark>Mcp-Param-Pipeline: image-generation</mark></span>
      </div>

      <div class="http-json http-json--packet">
        <div class="http-json-line">{</div>
        <div class="http-json-line http-json-line--indent"><em>"jsonrpc"</em>: <strong>"2.0"</strong>,</div>
        <div class="http-json-line http-json-line--indent"><em>"method"</em>: <mark>"tools/call"</mark>,</div>
        <div class="http-json-line http-json-line--indent"><em>"params"</em>: {</div>
        <div class="http-json-line http-json-line--indent-2"><em>"name"</em>: <mark>"hf.generate_image"</mark>,</div>
        <div class="http-json-line http-json-line--indent-2"><em>"arguments"</em>: {</div>
        <div class="http-json-line http-json-line--indent-3"><em>"prompt"</em>: <strong>"robot on a skateboard"</strong>,</div>
        <div class="http-json-line http-json-line--indent-3"><em>"pipeline"</em>: <mark>"image-generation"</mark></div>
        <div class="http-json-line http-json-line--indent-2">}</div>
        <div class="http-json-line http-json-line--indent">}</div>
        <div class="http-json-line">}</div>
      </div>
    </template>
  </section>
</template>
