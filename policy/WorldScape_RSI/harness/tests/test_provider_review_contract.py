"""Real provider transport/schema path with simulated HTTP responses."""
import copy,io,json,os
import pytest
from XPolicyLab.policy.WorldScape_RSI.harness.provider import ResponsesClient
from XPolicyLab.policy.WorldScape_RSI.harness.control_guard import policy_review_tool

BASE=[{'type':'function','function':{'name':'call_tool','description':'Local tools',
      'parameters':{'type':'object','properties':{},'required':[],'additionalProperties':False}}}]
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AUDIT_API_BASE","http://fixture.invalid/v1")
    monkeypatch.setenv("AUDIT_API_KEY","fixture")
    monkeypatch.delenv("R2SI_SESSION_RELEASE_PATH",raising=False)
    c=ResponsesClient(dict(model="fixture",effort="low",api_base_env="AUDIT_API_BASE",api_key_env="AUDIT_API_KEY"))
    payloads=[]
    def reply(request,**kwargs):
        body=json.loads(request.data);payloads.append(body)
        choice=body["tool_choice"]
        name=choice["name"] if choice.get("type")=="function" else choice["tools"][0]["name"]
        args={"decision":"takeover","progress":"wrong_direction","evidence":"The object moved away from the visible target"} if name=="review_policy_chunk" else {}
        return io.StringIO(json.dumps({"status":"completed","output":[{
            "type":"function_call","call_id":"c"+str(len(payloads)),"name":name,"arguments":json.dumps(args)}]}))
    c.opener.open=reply;c.payloads=payloads
    yield c
    c.close()
def test_regular_review_regular_preserves_sticky_context(client):
    messages=[{"role":"system","content":"Bounded fixture"},{"role":"user","content":"Initial public observation"}]
    a=client.complete(messages,BASE)["choices"][0]["message"]
    messages.extend([a,{"role":"tool","tool_call_id":a["tool_calls"][0]["id"],"content":"Action acknowledged"}])
    messages.append({"role":"user","content":"Fresh public state after WAM execution"})
    b=client.complete(messages,[policy_review_tool()])["choices"][0]["message"]
    assert json.loads(b["tool_calls"][0]["function"]["arguments"])["decision"]=="takeover"
    messages.extend([b,{"role":"tool","tool_call_id":b["tool_calls"][0]["id"],"content":"Suffix discarded"}])
    client.complete(messages,BASE)
    assert len(client.payloads)==3
    assert len({b["prompt_cache_key"] for b in client.payloads})==1
    assert client.tools==BASE
    assert client.payloads[0]["tools"]==client.payloads[1]["tools"]==client.payloads[2]["tools"]
    assert client.payloads[1]["tool_choice"]=={"type":"function","name":"review_policy_chunk"}
    assert client.payloads[0]["tool_choice"]["tools"]==[{"type":"function","name":"call_tool"}]
@pytest.mark.parametrize("mutation",["new_tool","review_schema","review_plus_tool"])
def test_arbitrary_tool_changes_remain_rejected(client,mutation):
    messages=[{"role":"user","content":"Initial observation"}];client.complete(messages,BASE)
    tools=[policy_review_tool()]
    if mutation=="new_tool":tools[0]["function"]["name"]="new_unadvertised_tool"
    elif mutation=="review_schema":tools[0]["function"]["parameters"]["properties"]["motion"]={"type":"string"}
    else:tools+=BASE
    with pytest.raises(RuntimeError,match="Tool definitions"):
        client.complete(messages,tools)
def test_review_cannot_start_an_episode(client):
    with pytest.raises(RuntimeError,match="existing Agent"):
        client.complete([{"role":"user","content":"No prior action"}],[policy_review_tool()])
def test_context_rewrite_remains_rejected(client):
    client.complete([{"role":"user","content":"Initial observation"}],BASE)
    with pytest.raises(RuntimeError,match="context changed"):
        client.complete([{"role":"user","content":"Rewritten observation"}],[policy_review_tool()])
