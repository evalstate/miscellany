<template>
  <section class="spec-timeline" aria-label="MCP specification transport and authorization evolution">
    <div class="spec-timeline__axis">
      <div class="spec-timeline__axis-spacer" aria-hidden="true" />
      <div class="spec-timeline__axis-track">
        <div class="spec-timeline__tick spec-timeline__tick--past spec-timeline__tick--col-1">
          <strong>2024-11</strong>
        </div>
        <div class="spec-timeline__tick spec-timeline__tick--col-2">
          <strong>2025-03</strong>
        </div>
        <div class="spec-timeline__tick spec-timeline__tick--col-3">
          <strong>2025-06</strong>
        </div>
        <div class="spec-timeline__tick spec-timeline__tick--col-4">
          <strong>2025-11</strong>
        </div>
        <div class="spec-timeline__tick spec-timeline__tick--now spec-timeline__tick--col-5">
          <strong>2026-07</strong>
        </div>
        <div class="spec-timeline__tick spec-timeline__tick--future spec-timeline__tick--col-6">
          <strong>...</strong>
        </div>
      </div>
    </div>

    <div class="spec-timeline__body">
      <div class="spec-timeline__lane-label spec-timeline__lane-label--transport">
        <span>Transports</span>
      </div>

      <div class="spec-timeline__lane spec-timeline__lane--transport">
        <div class="spec-timeline__grid-lines" aria-hidden="true" />
        <div class="spec-timeline__hotspots" aria-hidden="true">
          <div v-for="index in 6" :key="`transport-${index}`" :class="`spec-timeline__hotspot spec-timeline__hotspot--${index}`" />
        </div>

        <div class="spec-timeline__bar spec-timeline__bar--stdio">
          <strong>STDIO</strong>
        </div>
      </div>

      <div class="spec-timeline__lane-label spec-timeline__lane-label--remote">
        <span>Remote<br />Transports</span>
      </div>

      <div class="spec-timeline__lane spec-timeline__lane--remote">
        <div class="spec-timeline__grid-lines" aria-hidden="true" />
        <div class="spec-timeline__hotspots" aria-hidden="true">
          <div v-for="index in 6" :key="`remote-${index}`" :class="`spec-timeline__hotspot spec-timeline__hotspot--${index}`" />
        </div>

        <div class="spec-timeline__bar spec-timeline__bar--sse">
          <strong>SSE</strong>
        </div>

        <div class="spec-timeline__bar spec-timeline__bar--streamable">
          <strong>Streamable HTTP</strong>
        </div>

        <div class="spec-timeline__bar spec-timeline__bar--stateless">
          <strong>Stateless HTTP</strong>
        </div>
      </div>

      <div class="spec-timeline__lane-label spec-timeline__lane-label--auth">
        <span>Auth</span>
      </div>

      <div class="spec-timeline__lane spec-timeline__lane--auth">
        <div class="spec-timeline__grid-lines" aria-hidden="true" />
        <div class="spec-timeline__hotspots" aria-hidden="true">
          <div v-for="index in 6" :key="`auth-${index}`" :class="`spec-timeline__hotspot spec-timeline__hotspot--${index}`" />
        </div>

        <div class="spec-timeline__bar spec-timeline__bar--auth-as">
          <strong>OAuth AS</strong>
        </div>

        <div class="spec-timeline__bar spec-timeline__bar--auth-rs">
          <strong>OAuth Resource Server</strong>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.spec-timeline {
  container-type: size;
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  display: grid;
  --timeline-label-width: clamp(128px, 15cqw, 168px);
  --timeline-gap-x: clamp(0.78rem, 1.8cqw, 1.12rem);
  grid-template-rows: clamp(54px, 12cqh, 68px) 1fr;
  gap: clamp(0.72rem, 2.2cqh, 1rem);
  padding: clamp(1rem, 3.2cqh, 1.45rem);
  border: 1px solid var(--deck-border);
  border-radius: calc(var(--deck-radius) + 8px);
  background:
    radial-gradient(circle at 11% 24%, rgba(245, 164, 0, 0.12), transparent 26%),
    radial-gradient(circle at 82% 72%, rgba(106, 163, 247, 0.11), transparent 32%),
    linear-gradient(135deg, rgba(245, 164, 0, 0.055), transparent 44%),
    rgba(20, 22, 27, 0.84);
  box-shadow: var(--deck-shadow);
}

.spec-timeline::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(245, 164, 0, 0.034) 1px, transparent 1px),
    linear-gradient(90deg, rgba(245, 164, 0, 0.028) 1px, transparent 1px);
  background-size: 40px 40px;
  mask-image: radial-gradient(circle at 52% 48%, black, transparent 82%);
}

.spec-timeline > * {
  position: relative;
  z-index: 1;
}

.spec-timeline__axis {
  display: grid;
  grid-template-columns: var(--timeline-label-width) minmax(0, 1fr);
  column-gap: var(--timeline-gap-x);
  align-items: stretch;
}

.spec-timeline__axis-spacer {
  min-width: 0;
}

.spec-timeline__axis-track {
  min-width: 0;
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  align-items: stretch;
}

.spec-timeline__tick {
  min-width: 0;
  display: grid;
  align-content: center;
  position: relative;
  z-index: 3;
  margin: 0 clamp(0.22rem, 0.7cqw, 0.38rem);
  padding: clamp(0.42rem, 1.55cqh, 0.62rem) clamp(0.38rem, 1cqw, 0.62rem);
  border: 1px solid var(--deck-border);
  border-radius: calc(var(--deck-radius) + 5px);
  background:
    linear-gradient(180deg, rgba(245, 164, 0, 0.07), transparent 72%),
    rgba(11, 12, 15, 0.52);
  box-shadow: 0 14px 30px rgba(0, 0, 0, 0.18);
  transition:
    border-color 180ms ease,
    background 180ms ease,
    box-shadow 180ms ease,
    transform 180ms ease;
}

.spec-timeline__tick strong {
  color: var(--deck-text);
  font-size: clamp(1.05rem, 4.7cqh, 1.48rem);
  line-height: 1;
  letter-spacing: -0.06em;
  white-space: nowrap;
}

.spec-timeline__tick--now {
  border-color: rgba(255, 198, 73, 0.54);
  box-shadow:
    0 14px 30px rgba(0, 0, 0, 0.2),
    0 0 28px rgba(245, 164, 0, 0.12);
}

.spec-timeline__tick--future {
  border-color: rgba(106, 163, 247, 0.42);
  background:
    linear-gradient(180deg, rgba(106, 163, 247, 0.09), transparent 72%),
    rgba(11, 12, 15, 0.52);
}

.spec-timeline__body {
  min-height: 0;
  display: grid;
  grid-template-columns: var(--timeline-label-width) minmax(0, 1fr);
  grid-template-rows: minmax(0, 0.75fr) minmax(0, 2fr) minmax(0, 1fr);
  gap: clamp(0.62rem, 1.8cqh, 0.9rem) var(--timeline-gap-x);
}

.spec-timeline__lane-label {
  min-height: 0;
  display: grid;
  align-content: center;
  padding: 0.85rem;
  border: 1px solid var(--deck-border-2);
  border-radius: calc(var(--deck-radius) + 6px);
  background: rgba(11, 12, 15, 0.42);
}

.spec-timeline__lane-label span {
  color: var(--deck-text);
  font-size: clamp(0.86rem, 3.6cqh, 1.14rem);
  font-weight: 900;
  letter-spacing: 0.11em;
  text-transform: uppercase;
}

.spec-timeline__lane-label--transport {
  border-color: rgba(245, 164, 0, 0.36);
}

.spec-timeline__lane-label--remote {
  border-color: rgba(106, 163, 247, 0.35);
}

.spec-timeline__lane-label--auth {
  border-color: rgba(106, 163, 247, 0.35);
}

.spec-timeline__lane {
  position: relative;
  min-height: 0;
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  padding: clamp(0.7rem, 2.2cqh, 0.98rem) 0;
  border: 1px solid var(--deck-border);
  border-radius: calc(var(--deck-radius) + 7px);
  background: rgba(11, 12, 15, 0.24);
  overflow: hidden;
}

.spec-timeline__lane--transport {
  grid-template-rows: minmax(0, 1fr);
}

.spec-timeline__lane--remote {
  grid-template-rows: repeat(3, minmax(0, 1fr));
}

.spec-timeline__lane--auth {
  grid-template-rows: repeat(2, minmax(0, 1fr));
}

.spec-timeline__grid-lines {
  position: absolute;
  inset: 0;
  z-index: 0;
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  pointer-events: none;
}

.spec-timeline__grid-lines::before,
.spec-timeline__grid-lines::after {
  content: "";
}

.spec-timeline__grid-lines {
  background:
    linear-gradient(90deg, transparent calc(16.666% - 1px), rgba(185, 179, 165, 0.12) calc(16.666% - 1px), rgba(185, 179, 165, 0.12) calc(16.666% + 1px), transparent calc(16.666% + 1px)),
    linear-gradient(90deg, transparent calc(33.333% - 1px), rgba(185, 179, 165, 0.12) calc(33.333% - 1px), rgba(185, 179, 165, 0.12) calc(33.333% + 1px), transparent calc(33.333% + 1px)),
    linear-gradient(90deg, transparent calc(50% - 1px), rgba(185, 179, 165, 0.12) calc(50% - 1px), rgba(185, 179, 165, 0.12) calc(50% + 1px), transparent calc(50% + 1px)),
    linear-gradient(90deg, transparent calc(66.666% - 1px), rgba(185, 179, 165, 0.12) calc(66.666% - 1px), rgba(185, 179, 165, 0.12) calc(66.666% + 1px), transparent calc(66.666% + 1px)),
    linear-gradient(90deg, transparent calc(83.333% - 1px), rgba(106, 163, 247, 0.18) calc(83.333% - 1px), rgba(106, 163, 247, 0.18) calc(83.333% + 1px), transparent calc(83.333% + 1px));
}

.spec-timeline__hotspots {
  position: absolute;
  inset: 0;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
}

.spec-timeline__hotspot {
  min-width: 0;
  border-inline: 1px solid transparent;
  transition:
    background 160ms ease,
    border-color 160ms ease,
    box-shadow 160ms ease;
}

.spec-timeline__hotspot:hover {
  background:
    linear-gradient(180deg, rgba(255, 198, 73, 0.11), rgba(106, 163, 247, 0.075));
  border-color: rgba(255, 198, 73, 0.28);
  box-shadow: inset 0 0 28px rgba(255, 198, 73, 0.08);
}

.spec-timeline:has(.spec-timeline__hotspot--1:hover) .spec-timeline__hotspot--1,
.spec-timeline:has(.spec-timeline__tick--col-1:hover) .spec-timeline__hotspot--1,
.spec-timeline:has(.spec-timeline__hotspot--2:hover) .spec-timeline__hotspot--2,
.spec-timeline:has(.spec-timeline__tick--col-2:hover) .spec-timeline__hotspot--2,
.spec-timeline:has(.spec-timeline__hotspot--3:hover) .spec-timeline__hotspot--3,
.spec-timeline:has(.spec-timeline__tick--col-3:hover) .spec-timeline__hotspot--3,
.spec-timeline:has(.spec-timeline__hotspot--4:hover) .spec-timeline__hotspot--4,
.spec-timeline:has(.spec-timeline__tick--col-4:hover) .spec-timeline__hotspot--4,
.spec-timeline:has(.spec-timeline__hotspot--5:hover) .spec-timeline__hotspot--5,
.spec-timeline:has(.spec-timeline__tick--col-5:hover) .spec-timeline__hotspot--5,
.spec-timeline:has(.spec-timeline__hotspot--6:hover) .spec-timeline__hotspot--6,
.spec-timeline:has(.spec-timeline__tick--col-6:hover) .spec-timeline__hotspot--6 {
  background:
    linear-gradient(180deg, rgba(255, 198, 73, 0.13), rgba(106, 163, 247, 0.08));
  border-color: rgba(255, 198, 73, 0.32);
  box-shadow:
    inset 0 0 30px rgba(255, 198, 73, 0.09),
    0 0 28px rgba(245, 164, 0, 0.055);
}

.spec-timeline:has(.spec-timeline__hotspot--1:hover) .spec-timeline__tick--col-1,
.spec-timeline:has(.spec-timeline__tick--col-1:hover) .spec-timeline__tick--col-1,
.spec-timeline:has(.spec-timeline__hotspot--2:hover) .spec-timeline__tick--col-2,
.spec-timeline:has(.spec-timeline__tick--col-2:hover) .spec-timeline__tick--col-2,
.spec-timeline:has(.spec-timeline__hotspot--3:hover) .spec-timeline__tick--col-3,
.spec-timeline:has(.spec-timeline__tick--col-3:hover) .spec-timeline__tick--col-3,
.spec-timeline:has(.spec-timeline__hotspot--4:hover) .spec-timeline__tick--col-4,
.spec-timeline:has(.spec-timeline__tick--col-4:hover) .spec-timeline__tick--col-4,
.spec-timeline:has(.spec-timeline__hotspot--5:hover) .spec-timeline__tick--col-5,
.spec-timeline:has(.spec-timeline__tick--col-5:hover) .spec-timeline__tick--col-5,
.spec-timeline:has(.spec-timeline__hotspot--6:hover) .spec-timeline__tick--col-6,
.spec-timeline:has(.spec-timeline__tick--col-6:hover) .spec-timeline__tick--col-6 {
  border-color: rgba(255, 198, 73, 0.64);
  background:
    linear-gradient(180deg, rgba(245, 164, 0, 0.16), rgba(106, 163, 247, 0.06)),
    rgba(11, 12, 15, 0.64);
  box-shadow:
    0 16px 32px rgba(0, 0, 0, 0.22),
    0 0 26px rgba(245, 164, 0, 0.18);
  transform: translateY(-1px);
}

.spec-timeline__bar {
  position: relative;
  z-index: 2;
  min-width: 0;
  height: clamp(34px, 7.8cqh, 44px);
  align-self: center;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 clamp(0.26rem, 0.8cqw, 0.42rem);
  padding: 0 clamp(0.58rem, 1.5cqw, 0.86rem);
  border: 1px solid transparent;
  border-radius: 999px;
  box-shadow: 0 15px 28px rgba(0, 0, 0, 0.22);
  pointer-events: none;
}

.spec-timeline__bar strong {
  color: var(--deck-text);
  font-size: clamp(0.96rem, 3.45cqh, 1.42rem);
  font-weight: 950;
  line-height: 1;
  letter-spacing: -0.045em;
  white-space: nowrap;
}

.spec-timeline__bar--stdio {
  grid-column: 1 / 7;
  grid-row: 1;
  background:
    linear-gradient(90deg, rgba(245, 164, 0, 0.28), rgba(245, 164, 0, 0.13)),
    rgba(20, 22, 27, 0.9);
  border-color: rgba(255, 198, 73, 0.42);
}

.spec-timeline__bar--sse {
  grid-column: 1 / 2;
  grid-row: 1;
  background: rgba(240, 107, 90, 0.16);
  border-color: rgba(240, 107, 90, 0.45);
}

.spec-timeline__bar--streamable {
  grid-column: 2 / 5;
  grid-row: 2;
  background:
    linear-gradient(90deg, rgba(106, 163, 247, 0.28), rgba(106, 163, 247, 0.13)),
    rgba(20, 22, 27, 0.9);
  border-color: rgba(106, 163, 247, 0.46);
}

.spec-timeline__bar--stateless {
  grid-column: 5 / 7;
  grid-row: 3;
  background:
    linear-gradient(90deg, rgba(106, 163, 247, 0.34), rgba(245, 164, 0, 0.2)),
    rgba(20, 22, 27, 0.94);
  border-color: rgba(255, 198, 73, 0.52);
  box-shadow:
    0 15px 28px rgba(0, 0, 0, 0.24),
    0 0 32px rgba(106, 163, 247, 0.11);
}

.spec-timeline__bar--auth-as {
  grid-column: 2 / 3;
  grid-row: 1;
  background: rgba(245, 164, 0, 0.16);
  border-color: rgba(255, 198, 73, 0.42);
}

.spec-timeline__bar--auth-as strong {
  font-size: clamp(0.5rem, 2.05cqh, 0.76rem);
  letter-spacing: -0.04em;
}

.spec-timeline__bar--auth-rs {
  grid-column: 3 / 7;
  grid-row: 2;
  background:
    linear-gradient(90deg, rgba(106, 163, 247, 0.3), rgba(106, 163, 247, 0.13)),
    rgba(20, 22, 27, 0.92);
  border-color: rgba(106, 163, 247, 0.46);
}
</style>
