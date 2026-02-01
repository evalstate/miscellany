---

---
https://www.canva.com/design/DAG4mJK7aCA/Bgxwur2O9xc93XSET2fbzw/edit
MCP is the connectivity layer for a lot of our 

Sub-agents
Gradio Spaces (*1000)
HF MCP front-door and proxy
Remote MCP Callers
Remote MCP  transfer workloads

LSP Assumptions

Local, your Agent was like an IDE 
 - Context management would be via embedded resources

---

Stateless?


Initialize().

---

MRTR
STATELESS
Sessions are a side effect. 

Show Gradio Init skip code

---

HTTP 

Protocol should be Transport Agnostic. BUT --> design of the HTPT . Session handling is MCP Specific (e.g. it's in the HTTP Header rather than the envelope).

gRPC.

---

