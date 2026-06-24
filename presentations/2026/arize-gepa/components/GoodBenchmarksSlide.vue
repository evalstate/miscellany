<template>
  <div class="good-benchmarks">
    <section class="good-benchmarks__plot" aria-label="Held-out benchmark transfer from base prompt to best GEPA candidate">
      <div class="good-benchmarks__headline">
        <span>held-out OpenClaw issue labels · 78 rows · 3 runs each</span>
        <strong>67.7% → 72.5%</strong>
        <em>+4.8 pts transfer</em>
      </div>

      <GoodBenchmarksChart />
    </section>

    <aside class="good-benchmarks__metrics">
      <table>
        <thead>
          <tr><th>metric</th><th>base</th><th>best</th></tr>
        </thead>
        <tbody>
          <tr><td>GEPA score</td><td>67.7</td><td>72.5</td></tr>
          <tr><td>row exact</td><td>45.3</td><td>50.9</td></tr>
          <tr><td>row Jaccard</td><td>72.1</td><td>76.6</td></tr>
          <tr><td>micro F1</td><td>78.1</td><td>82.0</td></tr>
        </tbody>
      </table>

      <div class="good-benchmarks__checks">
        <article><strong>Representative</strong><span>Match density/distribution of data</span></article>
        <article><strong>Aligned</strong><span>scores what matters</span></article>
        <article><strong>Sensitive</strong><span>Detects Gains/Losses, Scales to model strength</span></article>
      </div>
    </aside>
  </div>
</template>

<script setup lang="ts">
import GoodBenchmarksChart from './GoodBenchmarksChart.vue'
</script>
