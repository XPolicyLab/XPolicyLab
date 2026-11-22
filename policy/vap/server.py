"""Policy-local exclusivity guard around the unmodified official WS server."""
import asyncio
from XPolicyLab.client_server.ws.model_server import PolicyServer, PolicyServerConfig
from XPolicyLab.setup_policy_server import parse_args_and_config
from XPolicyLab.policy.vap.model import Model
class ExclusiveServer(PolicyServer):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.owner=False
 async def _handle_connection(self,ws):
  if self.owner:
   await ws.close(code=1013,reason='one active client per VAP model');return
  self.owner=True
  try:await super()._handle_connection(ws)
  finally:self.owner=False
if __name__=='__main__':
 cfg=parse_args_and_config()
 model=Model(cfg)
 from XPolicyLab.policy.vap.server_acceleration import initialize
 model.executor.submit(initialize,model).result()
 server=ExclusiveServer(model,PolicyServerConfig(host=cfg['host'],port=int(cfg['port']),ws_ping_interval_s=cfg.get('ws_ping_interval_s',20.0),ws_ping_timeout_s=cfg.get('ws_ping_timeout_s',20.0)))
 asyncio.run(server.serve_forever())
