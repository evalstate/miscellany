<template>
  <svg viewBox="0 0 820 430" role="img" aria-label="Benchmark score distribution for base and GEPA optimized prompt">
    <defs>
      <linearGradient id="benchmarkDelta" x1="0" x2="1">
        <stop offset="0%" stop-color="#8ee8ff" />
        <stop offset="100%" stop-color="#ffb86b" />
      </linearGradient>
      <marker id="benchmarkArrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M 1 1 L 11 6 L 1 11 z" fill="#ffb86b" />
      </marker>
    </defs>

    <g class="good-benchmarks__grid">
      <line x1="110" x2="750" y1="380" y2="380" />
      <line x1="110" x2="750" y1="286" y2="286" />
      <line x1="110" x2="750" y1="193" y2="193" />
      <line x1="110" x2="750" y1="99" y2="99" />
      <text x="72" y="385">60%</text>
      <text x="72" y="291">65%</text>
      <text x="72" y="198">70%</text>
      <text x="72" y="104">75%</text>
    </g>

    <path class="good-benchmarks__delta" d="M 262 232 C 338 212 438 179 514 154" />
    <text class="good-benchmarks__improvement-label good-benchmarks__improvement-label--strong" x="392" y="153">+4.8 pts</text>
    <line class="good-benchmarks__mean good-benchmarks__mean--base" x1="188" x2="252" y1="235" y2="235" />
    <line class="good-benchmarks__mean good-benchmarks__mean--best" x1="528" x2="592" y1="145" y2="145" />

    <g class="good-benchmarks__dots good-benchmarks__dots--base">
      <circle cx="198" cy="247" r="7" />
      <circle cx="220" cy="248" r="7" />
      <circle cx="243" cy="211" r="7" />
    </g>
    <g class="good-benchmarks__dots good-benchmarks__dots--best">
      <circle cx="538" cy="158" r="7" />
      <circle cx="560" cy="126" r="7" />
      <circle cx="583" cy="151" r="7" />
    </g>

    <text class="good-benchmarks__label" x="220" y="414">Base prompt</text>
    <text class="good-benchmarks__label" x="560" y="414">Best GEPA candidate</text>
  </svg>
</template>
