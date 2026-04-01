at hugging face we host 1000s of MCP Servers mostly machine learning applications on our Spaces platform - so people can publish, tune, share and play with these applications. We provide access to these through the Hugging Face Hub.

We'll do a quick tour of a couple of a couple of features how we use MCP and Skills to help AI Developers and Users. 

The MCP Server is configurable with features for both security and of course token management.

For MCP Users, we have a couple of nice features -- that I' going to demonstrate and look at a couple of features of the MCP transport.

To make these ML applications more accessible, without too much token bloat *and* with decent interactivity we have a single "dynamic" tool which uses progressive disclosure as well as POST transport responses. We'll take a look at a basic demo.

So that works, we are efficient with the transport and we have decent progress notifications - by default we curate but that gives you access to dozens of spaces.

---

For us, this type of inference is important and why we are excited about the migration to Stateless - and better usage of the protocol. I'll share a bit of data here on what we're seeing (Orange/Black).

---

One are which has superseded and effectively replaced MCP via STDIO. Although MCP gives us a decent remote execution engine with sandboxes. I'm not going to fine tune a model live, but what we will do is look at using one of my favourite Skills - the HF MCP Tool Builder. 

Lets kick it off and then I'll talk about what's happening. 
OpenAPI -> Discovery -> Testing.

Now as an added bonus we can have that as a Subagent -- but first lets see if we can deploy that as an MCP Server.  So "fast-agent" the same tool I am using here locally for development and evals is able to deploy. One of the nice things about deploying to HF if you want to deploy subagents like this is that HF Accounts have inference credits and can do a lot with 

Check out `upskill` too -- if you want to do multi model skill optimisation and testing -- big improvements there, the first release shows effectively distilling information from models.






