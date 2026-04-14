# 通用的server-client

## /Simple_Remote_Infer/deploy/qwenpi_policy.py

- QwenPiServer: 一个用于示范的模型，拥有init和infer方法

将加载好的模型用`WebsocketPolicyServer`包裹，并指定端口即可

```python
model_server = WebsocketPolicyServer(model, port=8002)
model_server.serve_forever() # 开启监听
```

## ./websocket_client_policy.py

在`__main__()`中展示了如何创造一个假模型向真模型发送环境信息，只需要用`WebsocketClientPolicy`代替原有的模型即可