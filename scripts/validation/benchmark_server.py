"""Benchmark-only HTTP adapter; never use as a public deployment server."""
import argparse
import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--backend', choices=['native', 'onnx'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.backend == 'native':
        import torch
        from transformers import AutoModel, AutoProcessor
        torch.set_num_threads(a.threads)
        model = AutoModel.from_pretrained(a.model, trust_remote_code=True, local_files_only=True,
                                          dtype=torch.float32, attn_implementation='eager').float().cpu().eval()
        model.backbone_autocast_dtype = None
        processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)

        def predict(body):
            # The native Python model consumes local video paths. This adapter
            # materializes inline video solely for the localhost measurement.
            # It does not add video support to SGLang/vLLM production services.
            with TemporaryDirectory(prefix='valen-benchmark-video-') as directory:
                body = deepcopy(body)
                source = body.get('input', body.get('state'))
                messages = source if isinstance(source, list) else source.get('messages', []) if isinstance(source, dict) else []
                for message in messages:
                    for part in message.get('content', []) if isinstance(message.get('content'), list) else []:
                        key = 'video_url'
                        value = part.get(key)
                        url = value.get('url') if isinstance(value, dict) else value
                        if isinstance(url, str) and url.startswith('data:video/mp4;base64,'):
                            path = Path(directory) / 'video.mp4'
                            path.write_bytes(base64.b64decode(url.split(',', 1)[1], validate=True))
                            part[key] = {'url': str(path)} if isinstance(value, dict) else str(path)
                with torch.inference_mode():
                    return model.predict(body, processor=processor, execution='shared_state')
    else:
        from valen_sdk import Valen
        model = Valen.local(a.model, variant='fp32', local_files_only=True, num_threads=a.threads, model_name='valen')
        predict = model.predict

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, value, status=200, seconds=None):
            raw = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            if seconds is not None:
                self.send_header('X-Valen-Inference-Seconds', str(seconds))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.send({'status': 'ok'}) if self.path == '/health' else self.send({'error': 'not found'}, 404)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path not in ['/v1/decisions', '/v1/systemone']:
                    self.send({'error': 'not found'}, 404)
                    return
                field = 'input' if self.path.endswith('decisions') else 'state'
                if field not in body:
                    raise ValueError('Missing ' + field)
                start = time.perf_counter()
                result = predict(body)
                seconds = time.perf_counter() - start
                self.send(result, seconds=seconds)
            except (ValueError, TypeError, KeyError) as exc:
                self.send({'error': str(exc)}, 400)
            except Exception as exc:
                self.send({'error': type(exc).__name__ + ': ' + str(exc)}, 500)

    try:
        HTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
    finally:
        if a.backend == 'onnx':
            model.close()


if __name__ == '__main__':
    main()
