<!--
# The conversation metaphor

Turn taking example (can i steal a slide from the Anthropic website)

What's interesting is that a Tool result is representing the next User input to the conversation.
-->

<!-- We need to figure out perhaps there are different classes of Tool -->

<!--

So we might say "look, that's fine that we want to use the LLM to string together low-level API calls. While the LLM can be auite resilient.

-->
<!--
# The conversation metaphor

Turn taking example (can i steal a slide from the Anthropic website)

What's interesting is that a Tool result is representing the next User input to the conversation.
-->

<!-- We need to figure out perhaps there are different classes of Tool -->

<!--

So we might say "look, that's fine that we want to use the LLM to string together low-level API calls. While the LLM can be auite resilient.

-->
<!--

MCP Servers have lifted off as they have because of this. Tools are the closest thing to "just working", especially if they stick to plain text results. LLMs are quite forgiving.

All we need to have agreed is to use MCP!

Pleasing, surprising results can be delivered here.

-->
<!--
There's a couple of assumptions in the previous statement. Lets think abou this from the perspective of an Application developer integrating with MCP.

To date there are very few "MCP Native" clients; typically MCP Tool Support has been fairly straightforward to add for existing LLM Applications.

-->

<!-- >> \***Reference** LLMs get lost in multi-turn conversation - https://arxiv.org/pdf/2505.06120
-->

<!--


They turn us in to Prompt Engineers but it's hard to know what's going on. Models are sensitive to Tool Descriptions in different ways; if I want to construct a complex query do I put that in the tool description, or do I create a Tool that returns the instructions?


We all love tools, but they come with some drawbacks. They are context hungry. Mix is quite unpredictable.

If I know that I need to work with some data, then generating a Tool Call.

-->
<!--
# GPT-4.1

<center>

![w:1050](2025-05-22-gpt-4.1.png)

</center>

-->
<!--
A paper was released earlier this month by Microsoft Research, looking at the loss of reliability in multi-turn conversations.

The problem being that reliability will degrade significantly if we are relying on the LLM to string together tool calls.

they make the point that LLMs are often evaluated using "fully specified instructions", and that conversational interfaces are far less reliable. I think we all recognise the issue about context getting polluted with unnecessary information. 35% -->

<!-- Analysis of 200,000+ simulated conversations decomposes the performance degradation into two components: a minor loss in aptitude and a significant increase in unreliability. We find that LLMs often make assumptions in early turns and prematurely attempt to generate final solutions, on which they overly rely.

## Tools need to reduce their surface area. Straight API wraps work, but won't excel. Ideas like selecting a "Bouqet" of tools.

## User Interfaces needs to start . Ahead of time.

 -->

<!--

We also need to answer What can the Host/User do as a pair.

PICTURE VSCODE + GRADIO
PICTURE REPRESENTATION OF CONTENT TYPES:

For the code, it's possible to syntax highlight, but I'm still sending it to the LLM as `text/plain`.

MCP is a weird thing to work with (compared to traditional APIs). You're connected things which do Natural Language Processing, to things which produce unpredictable answers.

So for example "Can I show it might mean having image handling libraries, or an audio renderer. Can the LLM understand it might mean tokenizing it? And those combinations may, or may not be valid.

An example might be doing Syntax Highlighting on Source Code, but text/plain, markdown and modern formatters already do this job anyway! Markdown is self-extensible with things ```mermaid and so on.

I might be able to render a video, or just save the content to disk. It's kind of up to me to figure out what to do.

VSCode connecting to Gradio. It can render images.
-->
<!-- we know if it's bits and bytes or characters, and we know the shape of those parts because of the MIMEType
<!-- one way of thinking about Prompts is as a _user driven tool_ -->

<!-- whilst it's amazing to watch LLMs discover the world around them via Tools, for a lot of Servers it is unnecessary -->
<!-- as previously mentioned, a well designed MCP server that can offer Prompts optimied for intent will get us better results and parsimonious use of Tools -->

<!-- more sophisticated clients that enable you to compose prompts with live resources etc will be welcome -->
<!--
Prompts are more powerful that it first appears.
1) You can put arbitrary (and live text in them), injecting data directly in to the context without Tool Call overhead or unnecessary conversation turns.
2) Because they are conversational pairs, they can provide in-context learning for the follow-up conversation.
3) I could construct a prompt
Can be used for in-context learning -->
<!--

But this _does_ require the Client application to know whether resources are subscribed.
Even though the MCP Server can inject whatever content it wants in to

- Example 1: Lint Errors
- Example 2: mcp-webcam Subscription
- Example 3: User Interaction
- Example 4: Subscribe to a Log File tail - trigger a completion - "USER: Send an alert to the on-call engineer if anything anomalous is in the logs?"
- Trigger a Regneration.


-->
<!-- There is a PulseMCP video where we did a couple of those things

MCP-Webcam will have this capability soon!

-->

<!-- This is where extra preparation time for the talk has saved me. Twixt my overambition and actual capabilities  -->
<!-- it actually ends up being more work than adjusting the protocol

this also makes it possible for MCP Server developers to coordinate amongst themselves (e.g. managing name clashes and so on).
-->
<!-- Search data. Manipulate that dataset -->

<!--

A lot of first instincts will be to modify the specification somehow. I have "use-case _x_, it might be generally useful, so I'll build something specific around it

It would be a shame to keep adding things to the protocol when extension mechansims. We keep bodging and adding edge cases in to Tools because that's where the light is.

-->

<!--

Example. Getting a Customer Record as a result with a known URI Scheme, would allow a subscription to be made, or navigate to say "sales information", "NBA" etc.

Example. Long-running task returning an initial frame with later content being available as a Resource.

Prompts for IDE/API Servers for application building including https:// resources (e.g. llms.txt) to be resolved by the Host for inclusion in context during code generation.

Commonly agreed URI scheme for Home Automation sensors and controls would allow integrators to use standardised libraries to build all kinds of embedded agent based systems with low effort.

UX generation patterns (such as Claude Artifacts) delivered by commonly agreed schemes would allow rich clients (Claude Desktop, Sage, 5ire and others) to use standard SDKs to show rich User Interfaces from MCP Servers that supported the scheme.
-->

<!--
What might slow that down?
Knowledge
-->

<!--
# Futures

## At least one SDK for Client application developers to use.
## Some Specification refinements (the ability to use without Embedding e.g.)
## Tying up Roots and Resources (especially with Remote MCP Servers)
## Sampling would be better if it supported Resources.
## Add URI sceheme to the Registry
-->
<!--

We can either have every Client know about every Server, or every Server know specifically. MCP is still useful in that scenario; because it gives you standards, and using basic tools stuff will probably still work.

We know this -- it's just restating the MxN problem.

There's a lot of good things that can be achieved just by Client applications providing richer constructs for assembling context using the primitives, and more transparently exposing Resources and Prompts to be used.

-->

<!--
Inversion of control


What we are always trying to do is to manage the complexity of the real world, different competing software systems. I hope that instinctively we can keep the core protocol simple and EXTENSIBLE. MCP clients should be 50 lines of code, Servers should expect to work everywhere. The danger is that powerful parts of the protocol go dark, and less optimal solutions are . Push in or out of the protocol. Push in our out of _this part_ of the protocol. Barrier to entry.

-->

<!-->

The MCP Server Developer can provide some help here; if the content is or isn't intended for the LLM, or prioritised for User consumption.

If you want - and the server and uri scheme support it - you can subscribe for live updates to them, specify a templating scheme and use completions for discovery - but these things are all optional if you need them. I'd encourage people to use those capabilities, and think it good practice to have an integrated scheme, but
-->
