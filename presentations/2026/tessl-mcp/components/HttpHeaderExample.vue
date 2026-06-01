<template>
  <section class="http-request-panel deck-panel">
    <h2>MCP over HTTP</h2>

    <div class="http-tool-panel">
      <span>tool definition</span>
      <div class="http-code-row">
        <em>name</em><mark>hf.hub_search</mark>
      </div>
      <div class="http-code-row">
        <em>description</em><strong>Search Hub models and datasets</strong>
      </div>
    </div>

    <div class="http-request-line">
      <strong>POST /mcp/ HTTP/1.1</strong>
      <span>Host: hf.co/mcp</span>
      <span><mark>Mcp-Method: tools/call</mark></span>
      <span><mark>Mcp-Name: hf.hub_search</mark></span>
    </div>

    <div class="http-json">
      <div class="http-code-row">
        <em>jsonrpc</em><strong>2.0</strong>
      </div>
      <div class="http-code-row">
        <em>method</em><mark>tools/call</mark>
      </div>
      <div class="http-code-row">
        <em>params.name</em><mark>hf.hub_search</mark>
      </div>
      <div class="http-code-row">
        <em>arguments.query</em><strong>image generation models</strong>
      </div>
      <div class="http-code-row">
        <em>arguments.type</em><strong>model</strong>
      </div>
    </div>
  </section>
</template>
