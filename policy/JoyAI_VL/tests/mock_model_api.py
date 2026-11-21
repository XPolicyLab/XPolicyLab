"""Synthetic API fixture for interface validation; no benchmark task logic."""
import argparse,base64,io,json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from PIL import Image

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        pictures=[]
        for message in request['messages']:
            if isinstance(message.get('content'),list):
                for item in message['content']:
                    if item.get('type')=='image_url':
                        raw=base64.b64decode(item['image_url']['url'].split(',',1)[1])
                        with Image.open(io.BytesIO(raw)) as image:
                            assert image.mode=='RGB'
                            pictures.append(list(image.size))
        assert pictures, 'API request must carry camera images'
        assert request['tools'] and request['model']=='interface-check'
        print(json.dumps({'event':'request','images':pictures,'messages':len(request['messages'])}),flush=True)
        response={'id':'interface-check','object':'chat.completion','choices':[{'index':0,'finish_reason':'tool_calls','message':{
            'role':'assistant','content':'','tool_calls':[{'id':'check-call','type':'function','function':{'name':'python','arguments':json.dumps({'code':"robot.find(text='object'); robot.look(); robot.wait(3)"})}}]}}]}
        data=json.dumps(response).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=19400);args=parser.parse_args()
    ThreadingHTTPServer(('0.0.0.0',args.port),Handler).serve_forever()
