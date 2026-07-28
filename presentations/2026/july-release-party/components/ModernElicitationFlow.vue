<script setup lang="ts">
import { computed, ref } from "vue";
import { Box, Check, Monitor, Server, UserRound } from "@lucide/vue";
import { useTimedStoryboard } from "../composables/useTimedStoryboard";

type Phase =
  | "request-one"
  | "input-required"
  | "confirm-cost"
  | "confirmation-accepted"
  | "retry-request"
  | "complete-result";
type Direction = "outbound" | "return";
type RequestStatus = "not-started" | "open" | "closed";
type ConfirmationStatus = "hidden" | "awaiting" | "accepted";
type SandboxStatus = "absent" | "creating" | "ready";
type ServerStatus = "idle" | "handling-request" | "creating-sandbox" | "ready";

type Frame = {
  phase: Phase;
  title: string;
  detail: string;
  packet?: string;
  requestId?: 1 | 2;
  direction?: Direction;
  requestOne: RequestStatus;
  requestTwo: RequestStatus;
  confirmation: ConfirmationStatus;
  sandbox: SandboxStatus;
  server: ServerStatus;
  duration: number;
};

const frames = computed<readonly Frame[]>(() => [
  {
    phase: "request-one",
    title: "1 · Create a GPU sandbox",
    detail: "The Client calls hf.create_sandbox. This operation will cost $0.40 per hour.",
    packet: "tools/call",
    requestId: 1,
    direction: "outbound",
    requestOne: "open",
    requestTwo: "not-started",
    confirmation: "hidden",
    sandbox: "absent",
    server: "handling-request",
    duration: 1900,
  },
  {
    phase: "input-required",
    title: "2 · Ask before spending",
    detail: "The Server returns input_required with a keyed elicitation. Request 1 is finished.",
    packet: "input_required",
    requestId: 1,
    direction: "return",
    requestOne: "closed",
    requestTwo: "not-started",
    confirmation: "hidden",
    sandbox: "absent",
    server: "idle",
    duration: 2500,
  },
  {
    phase: "confirm-cost",
    title: "3 · Show a meaningful confirmation",
    detail: "The Client presents the Server's confirmation request to the person.",
    requestOne: "closed",
    requestTwo: "not-started",
    confirmation: "awaiting",
    sandbox: "absent",
    server: "idle",
    duration: 5000,
  },
  {
    phase: "confirmation-accepted",
    title: "4 · Cost approved",
    detail: "The accepted form becomes inputResponses[\"confirm_cost\"].",
    requestOne: "closed",
    requestTwo: "not-started",
    confirmation: "accepted",
    sandbox: "absent",
    server: "idle",
    duration: 1600,
  },
  {
    phase: "retry-request",
    title: "5 · Retry with the confirmation",
    detail: "Request 2 carries the original arguments, accepted response, and exact opaque state.",
    packet: "tools/call + confirmation",
    requestId: 2,
    direction: "outbound",
    requestOne: "closed",
    requestTwo: "open",
    confirmation: "accepted",
    sandbox: "creating",
    server: "creating-sandbox",
    duration: 2600,
  },
  {
    phase: "complete-result",
    title: "6 · Sandbox created",
    detail: "The Server completes request 2 only after the person has approved the cost.",
    packet: "complete",
    requestId: 2,
    direction: "return",
    requestOne: "closed",
    requestTwo: "closed",
    confirmation: "accepted",
    sandbox: "ready",
    server: "ready",
    duration: 5000,
  },
]);

const activated = ref(false);
const { active, animationKey, isRunning, play } = useTimedStoryboard<Frame>(
  frames,
  { defaultDuration: 1800, endDelay: 650 },
);

function run() {
  activated.value = true;
  play(true);
}
</script>

<template>
  <section
    class="sandbox-elicit"
    :class="[
      active?.phase && `is-${active.phase}`,
      { 'is-running': isRunning },
    ]"
    aria-label="Modern elicitation confirming the cost of a GPU sandbox"
    role="button"
    tabindex="0"
    title="Click to run the paid sandbox confirmation"
    @click.stop="run"
    @keydown.enter.prevent="run"
    @keydown.space.prevent="run"
  >
    <header class="sandbox-elicit__header">
      <div>
        <span>practical example · ephemeral workflow</span>
        <strong>{{ active?.title ?? "Click to create a GPU sandbox" }}</strong>
      </div>
      <p>{{ active?.detail ?? "A person must approve the hourly cost before creation." }}</p>
      <div class="sandbox-elicit__price">$0.40 / hour</div>
    </header>

    <div class="sandbox-elicit__stage">
      <div class="sandbox-elicit__route" aria-hidden="true" />

      <article
        class="sandbox-elicit__actor sandbox-elicit__actor--person"
        :class="{ 'is-active': active?.confirmation !== 'hidden' }"
      >
        <UserRound :stroke-width="2.2" />
        <div><span>person</span><strong>Shaun</strong></div>
      </article>

      <article class="sandbox-elicit__actor sandbox-elicit__actor--client">
        <Monitor :stroke-width="2.2" />
        <div><span>client</span><strong>MCP Client</strong></div>
      </article>

      <article
        class="sandbox-elicit__actor sandbox-elicit__actor--server"
        :class="{ 'is-active': active?.server === 'handling-request' || active?.server === 'creating-sandbox' || active?.server === 'ready' }"
      >
        <Server :stroke-width="2.2" />
        <div>
          <span>server</span>
          <strong>Sandbox Service</strong>
          <small v-if="active?.server === 'handling-request'">handling id 1</small>
          <small v-else-if="active?.server === 'creating-sandbox'">creating sandbox</small>
          <small v-else-if="active?.server === 'ready'">sbx-7f3c · ready</small>
          <small v-else>idle · no open request</small>
        </div>
        <Box v-if="active?.sandbox === 'ready'" class="sandbox-elicit__box" :stroke-width="2.2" />
      </article>

      <div
        v-if="active?.direction"
        :key="animationKey"
        class="sandbox-elicit__packet"
        :class="`sandbox-elicit__packet--${active.direction}`"
      >
        <strong>{{ active.packet }}</strong>
        <span>id {{ active.requestId }}</span>
      </div>

      <div
        v-if="active && active.confirmation !== 'hidden'"
        :key="`confirm-${animationKey}`"
        class="sandbox-elicit__confirmation"
        :class="{ 'is-accepted': active?.confirmation === 'accepted' }"
      >
        <header>
          <span>confirmation required</span>
          <strong v-if="active.confirmation === 'accepted'"><Check /> approved</strong>
        </header>
        <h3>Create this GPU sandbox?</h3>
        <p><strong>t4-small</strong> will cost <mark>$0.40 per hour</mark>.</p>
        <div>
          <span>Cancel</span>
          <strong>{{ active.confirmation === "accepted" ? "Confirmed" : "Create sandbox" }}</strong>
        </div>
      </div>
    </div>

    <footer class="sandbox-elicit__requests">
      <article :class="`is-${active?.requestOne ?? 'not-started'}`">
        <span>request 1</span>
        <strong>tools/call · id 1</strong>
        <small>{{ active?.requestOne ?? "not-started" }}</small>
      </article>
      <div class="sandbox-elicit__handoff">
        <span>new id</span>
        <strong>→</strong>
      </div>
      <article :class="`is-${active?.requestTwo ?? 'not-started'}`">
        <span>request 2</span>
        <strong>tools/call · id 2</strong>
        <small>{{ active?.requestTwo ?? "not-started" }}</small>
      </article>
    </footer>
  </section>
</template>

<style scoped>
.sandbox-elicit {
  container-type: size;
  width: 100%;
  height: 100%;
  min-height: 0;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  gap: clamp(0.65rem, 2.2cqh, 0.9rem);
  padding: clamp(0.86rem, 2.6cqh, 1.12rem);
  color: var(--deck-text);
  cursor: pointer;
  outline: none;
}

.sandbox-elicit:focus-visible {
  box-shadow: inset 0 0 0 2px var(--deck-accent-hi);
}

.sandbox-elicit__header {
  display: grid;
  grid-template-columns: minmax(0, 0.78fr) minmax(0, 1.1fr) auto;
  gap: 1rem;
  align-items: end;
}

.sandbox-elicit__header div:first-child > span {
  display: block;
  color: var(--deck-accent-hi);
  font: 800 0.54rem/1 var(--deck-font-mono);
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.sandbox-elicit__header div:first-child > strong {
  display: block;
  margin-top: 0.3rem;
  color: var(--deck-text);
  font-size: clamp(1.18rem, 4.9cqh, 1.62rem);
  letter-spacing: -0.05em;
}

.sandbox-elicit__header p {
  margin: 0;
  color: var(--deck-muted);
  font-size: clamp(0.76rem, 3.1cqh, 0.94rem);
  line-height: 1.28;
  text-align: right;
}

.sandbox-elicit__price {
  padding: 0.48rem 0.66rem;
  color: var(--deck-text);
  border: 1px solid rgba(255, 198, 73, 0.64);
  border-radius: 999px;
  background: var(--deck-highlight);
  font: 850 0.72rem/1 var(--deck-font-mono);
  white-space: nowrap;
}

.sandbox-elicit__stage {
  position: relative;
  min-height: 0;
  overflow: hidden;
  border: 1px solid var(--deck-border);
  border-radius: calc(var(--deck-radius) + 8px);
  background:
    radial-gradient(circle at 88% 18%, rgba(255, 198, 73, 0.1), transparent 28%),
    rgba(243, 244, 246, 0.42);
}

.sandbox-elicit__route {
  position: absolute;
  left: 15%;
  right: 15%;
  top: 34%;
  height: 4px;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(106, 163, 247, 0.2), rgba(255, 198, 73, 0.38));
}

.sandbox-elicit__actor {
  position: absolute;
  z-index: 2;
  top: 17%;
  width: 24%;
  min-width: 150px;
  min-height: 88px;
  display: flex;
  align-items: center;
  gap: 0.62rem;
  padding: 0.78rem 0.86rem;
  border: 1px solid var(--deck-border-2);
  border-radius: calc(var(--deck-radius) + 4px);
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.1);
}

.sandbox-elicit__actor--person { left: 3%; }
.sandbox-elicit__actor--client { left: 38%; }
.sandbox-elicit__actor--server { right: 3%; }

.sandbox-elicit__actor > svg {
  flex: none;
  width: 1.95rem;
  height: 1.95rem;
  color: var(--deck-muted);
}

.sandbox-elicit__actor span,
.sandbox-elicit__actor small {
  display: block;
  color: var(--deck-dim);
  font: 700 0.5rem/1 var(--deck-font-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.sandbox-elicit__actor strong {
  display: block;
  margin-top: 0.24rem;
  color: var(--deck-text);
  font-size: 1rem;
}

.sandbox-elicit__actor small {
  margin-top: 0.28rem;
  color: var(--deck-accent-hi);
}

.sandbox-elicit__actor.is-active {
  border-color: rgba(255, 198, 73, 0.82);
  box-shadow:
    0 8px 22px rgba(15, 23, 42, 0.1),
    0 0 0 4px rgba(255, 198, 73, 0.14);
}

.sandbox-elicit__box {
  margin-left: auto;
  color: var(--deck-ok) !important;
}

.sandbox-elicit__packet {
  position: absolute;
  z-index: 6;
  top: 30%;
  padding: 0.34rem 0.5rem;
  color: var(--deck-text);
  border: 1px solid rgba(255, 198, 73, 0.72);
  border-radius: 999px;
  background: var(--deck-highlight);
  box-shadow: 0 5px 16px rgba(180, 83, 9, 0.2);
  font-family: var(--deck-font-mono);
  white-space: nowrap;
}

.sandbox-elicit__packet strong { font-size: 0.66rem; }
.sandbox-elicit__packet span { margin-left: 0.34rem; font-size: 0.54rem; }
.sandbox-elicit__packet--outbound { animation: sandbox-packet-out 1050ms ease both; }
.sandbox-elicit__packet--return { animation: sandbox-packet-back 1050ms ease both; }

.sandbox-elicit__confirmation {
  position: absolute;
  z-index: 8;
  left: 7%;
  bottom: 4%;
  width: 49%;
  padding: 0.92rem 1rem;
  border: 1px solid rgba(255, 198, 73, 0.66);
  border-radius: calc(var(--deck-radius) + 5px);
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.16);
  animation: sandbox-confirm-in 420ms ease both;
}

.sandbox-elicit__confirmation > header,
.sandbox-elicit__confirmation > div {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.6rem;
}

.sandbox-elicit__confirmation header span {
  color: var(--deck-accent-hi);
  font: 800 0.6rem/1 var(--deck-font-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.sandbox-elicit__confirmation header strong {
  display: flex;
  align-items: center;
  gap: 0.2rem;
  color: var(--deck-ok);
  font-size: 0.76rem;
}

.sandbox-elicit__confirmation header svg {
  width: 0.8rem;
  height: 0.8rem;
}

.sandbox-elicit__confirmation h3 {
  margin: 0.58rem 0 0;
  font-size: 1.28rem;
}

.sandbox-elicit__confirmation p {
  margin: 0.34rem 0 0.68rem;
  color: var(--deck-muted);
  font-size: 0.9rem;
}

.sandbox-elicit__confirmation mark {
  padding: 0.04rem 0.18rem;
  color: var(--deck-text);
  border-radius: var(--deck-radius-sm);
  background: var(--deck-highlight);
}

.sandbox-elicit__confirmation > div span,
.sandbox-elicit__confirmation > div strong {
  padding: 0.38rem 0.56rem;
  border-radius: var(--deck-radius-sm);
  font-size: 0.78rem;
}

.sandbox-elicit__confirmation > div span {
  color: var(--deck-muted);
  border: 1px solid var(--deck-border);
}

.sandbox-elicit__confirmation > div strong {
  color: var(--deck-text);
  background: var(--deck-highlight);
}

.sandbox-elicit__confirmation.is-accepted {
  border-color: rgba(5, 150, 105, 0.62);
  background: rgba(236, 253, 245, 0.96);
}

.sandbox-elicit__requests {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 0.66rem;
  align-items: stretch;
}

.sandbox-elicit__requests article {
  position: relative;
  padding: 0.6rem 0.7rem;
  border: 1px solid var(--deck-border);
  border-radius: var(--deck-radius);
  background: rgba(255, 255, 255, 0.68);
  opacity: 0.46;
}

.sandbox-elicit__requests article.is-open,
.sandbox-elicit__requests article.is-closed { opacity: 1; }
.sandbox-elicit__requests article.is-open { border-color: rgba(106, 163, 247, 0.62); }
.sandbox-elicit__requests article.is-closed { border-color: rgba(255, 198, 73, 0.58); }

.sandbox-elicit__requests article > span,
.sandbox-elicit__requests article > small {
  display: block;
  color: var(--deck-dim);
  font: 700 0.48rem/1 var(--deck-font-mono);
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.sandbox-elicit__requests article > strong {
  display: block;
  margin-top: 0.28rem;
  font: 750 0.68rem/1 var(--deck-font-mono);
}

.sandbox-elicit__requests article > small {
  position: absolute;
  top: 0.52rem;
  right: 0.62rem;
}

.sandbox-elicit__handoff {
  display: grid;
  place-items: center;
  color: var(--deck-accent-hi);
}

.sandbox-elicit__handoff span {
  font: 700 0.4rem/1 var(--deck-font-mono);
  text-transform: uppercase;
}

.sandbox-elicit__handoff strong { font-size: 1.1rem; }

@keyframes sandbox-packet-out {
  from { left: 48%; opacity: 0; }
  to { left: 77%; opacity: 1; }
}

@keyframes sandbox-packet-back {
  from { left: 77%; opacity: 0; }
  to { left: 48%; opacity: 1; }
}

@keyframes sandbox-confirm-in {
  from { opacity: 0; transform: translateY(8px) scale(0.97); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
</style>
