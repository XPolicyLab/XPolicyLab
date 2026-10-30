"""Inference clients have no robot, simulator, filesystem or service Tools."""
import json,os,urllib.request,urllib.error,uuid
from pathlib import Path
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import AzureAgentClient,InfrastructureFailure

class JsonBridgeClient:
    def __init__(self,config):
        self.config=dict(config)
        if self.config.get('kind')!='roborsi_inference_only':
            raise InfrastructureFailure('Legacy robot-toolkit bridge rejected. Use the v1.7 inference-only bridge or the official API client.')
        self.token=Path(config['key_file']).read_text().strip() if config.get('key_file') else os.environ.get(config.get('key_env','ROBORSI_BRIDGE_TOKEN'),'')
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def complete(self,messages,tools):
        body={'model':self.config['model'],'effort':self.config.get('effort','medium'),'messages':messages,'tools':tools}
        headers={'Content-Type':'application/json'}
        if self.token:headers['Authorization']='Bearer '+self.token
        req=urllib.request.Request(self.config['url'],data=json.dumps(body).encode(),headers=headers)
        try:
            with self.opener.open(req,timeout=None) as response:result=json.load(response)
        except urllib.error.HTTPError as error:
            # Preserve the provider's classification for fresh-session recovery.
            # Never retry inference here: the current robot attempt must end first.
            raw=error.read(8192).decode('utf-8',errors='replace')
            try:
                detail=json.loads(raw).get('error','')
                if isinstance(detail,dict):
                    detail=' '.join(str(detail.get(k,'')) for k in ('type','code','message'))
            except (ValueError,AttributeError):detail=''
            raise InfrastructureFailure(f'Inference bridge request failed: HTTP {error.code}: {str(detail)[:2000]}') from None
        except Exception as error:raise InfrastructureFailure('Inference bridge request failed: '+type(error).__name__) from None
        if result.get('policy')!='inference-only-v1' or result.get('external_tool_calls')!=0:raise InfrastructureFailure('Bridge did not attest observation-only inference')
        if 'choices' not in result:raise InfrastructureFailure('Bridge returned no completion')
        return result

def build_client(env):
    if env.get('ROBORSI_PROVIDER_FILE'):return JsonBridgeClient(json.loads(Path(env['ROBORSI_PROVIDER_FILE']).read_text()))
    client=AzureAgentClient.from_env(env)
    # No arbitrary deadline on a reasoning turn. Connection failures still raise
    # infrastructure errors and upstream retry classification remains available.
    client.timeout_s=None;client.hard_timeout_s=0
    return client
