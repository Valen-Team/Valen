"""Serial, same-worker CPU benchmark. See docs/backend-validation.md.

Config supplies existing cached model commands; this tool never installs anything.
Each backend/modality gets a fresh process. Cold means process-cold, not disk-cold.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import threading
import time
import urllib.request

from check_backends import check, post, prepared_cases, validate


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def atomic_write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def memory_monitor(pid, stop, memory):
    # Linux VmHWM is the process RSS high-water mark, including file-backed RSS.
    while not stop.wait(0.025):
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmHWM:'):
                    memory['peak_rss_bytes'] = max(memory.get('peak_rss_bytes', 0), int(line.split()[1]) * 1024)
        except FileNotFoundError:
            break


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runs', type=int, default=7)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.runs < 2 or a.warmup < 1:
        p.error('At least two measured requests and one warmup required')
    config = json.loads(a.config.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    fixture_cases, _ = prepared_cases(config['image'], config['video'])
    fixtures = {name: body for name, body in fixture_cases if name in ['text', 'image', 'video']}
    identity = hashlib.sha256(a.config.read_bytes()).hexdigest()
    report_path = a.output / 'performance.json'
    report = {'schema_version': 1, 'completed': False, 'passed': False,
              'validation_kind': 'real_model_cpu_http', 'date_utc': datetime.now(timezone.utc).isoformat(),
              'config_sha256': identity, 'hardware': {'host': socket.gethostname(), 'platform': platform.platform(),
              'machine': platform.machine(), 'cpu': next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), 'unknown'),
              'affinity': sorted(os.sched_getaffinity(0))}, 'runs': a.runs, 'warmup_requests': a.warmup,
              'concurrency': 1, 'cold_definition': 'fresh server process; OS page cache not cleared',
              'percentile_method': 'linear interpolation on sorted samples',
              'memory_definition': 'server process VmHWM RSS; includes resident mapped pages; not model file size or GPU VRAM',
              'client_timing': 'localhost JSON encode + HTTP + server + JSON decode; media fixture bytes already prepared',
              'server_timing': 'native/onnx predict wall time including compiler and preprocessing; GGUF not exposed, null',
              'input_sha256': {name: hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() for name, body in fixtures.items()},
              'media_sha256': {name: hashlib.sha256(Path(config[name]).read_bytes()).hexdigest() for name in ['image', 'video']},
              'versions': config['versions'], 'records': []}
    if report_path.exists():
        if not a.resume:
            raise RuntimeError('Existing report: inspect it and use --resume to avoid duplicate work')
        old = json.loads(report_path.read_text())
        assert old['config_sha256'] == identity and old['runs'] == a.runs and old['warmup_requests'] == a.warmup
        # Never merge different machine allocations into a same-worker comparison.
        assert old['hardware'] == report['hardware'], 'Resume must use the same worker/hardware'
        report = old
        if old['completed']:
            raise SystemExit(0 if old['passed'] else 1)
    atomic_write(report_path, report)
    for backend in config['backends']:
        for modality, body in fixtures.items():
            if any(r['backend'] == backend['name'] and r['modality'] == modality for r in report['records']):
                continue
            row = {'backend': backend['name'], 'modality': modality, 'model_size': '0.8B',
                   'precision': backend['precision'], 'threads': config['threads'],
                   'model_provenance': backend['model_provenance'], 'media_budget': config['media_budget'], 'passed': False}
            print('START', backend['name'], modality, flush=True)
            log_path = a.output / (backend['name'] + '-' + modality + '.log')
            stop, memory = threading.Event(), {}
            with log_path.open('w') as log:
                started = time.perf_counter()
                proc = subprocess.Popen(backend['command'], env=os.environ | backend.get('env', {}), stdout=log, stderr=subprocess.STDOUT)
                monitor = threading.Thread(target=memory_monitor, args=(proc.pid, stop, memory), daemon=True)
                monitor.start()
                try:
                    deadline = time.monotonic() + 240
                    while time.monotonic() < deadline:
                        if proc.poll() is not None:
                            raise RuntimeError('Server exited; inspect ' + log_path.name)
                        try:
                            with urllib.request.urlopen(backend['health_url'], timeout=1) as response:
                                if response.status == 200:
                                    break
                        except Exception:
                            time.sleep(0.1)
                    else:
                        raise TimeoutError('Server readiness deadline')
                    row['startup_to_ready_seconds'] = time.perf_counter() - started
                    status, first, cold, cold_inference = post(backend['base_url'], body)
                    assert status == 200, first
                    validate(body, first)
                    row.update(cold_request_seconds=cold, cold_server_predict_seconds=cold_inference,
                               response=first, input_tokens=first['usage']['input_tokens'])
                    for _ in range(a.warmup):
                        status, response, *_ = post(backend['base_url'], body)
                        assert status == 200
                        validate(body, response)
                    e2e, inference = [], []
                    for _ in range(a.runs):
                        status, response, seconds, server_seconds = post(backend['base_url'], body)
                        assert status == 200
                        validate(body, response)
                        assert response['answers'] == first['answers'], 'Repeat result changed'
                        e2e.append(seconds)
                        if server_seconds is not None:
                            inference.append(server_seconds)
                    row.update(warm_samples_seconds=e2e, warm_p50_seconds=statistics.median(e2e), warm_p95_seconds=percentile(e2e, .95),
                               server_predict_samples_seconds=inference or None,
                               server_predict_p50_seconds=statistics.median(inference) if inference else None,
                               server_predict_p95_seconds=percentile(inference, .95) if inference else None, passed=True)
                    if modality == 'video':
                        regression = check(backend['base_url'], config['image'], config['video'])
                        atomic_write(a.output / (backend['name'] + '-contract.json'), regression)
                        # Cookbook CLI checks may use this final loaded ONNX service.
                        for task in backend.get('after_checks', []):
                            result = subprocess.run(task['command'], env=os.environ | task.get('env', {}), capture_output=True, text=True, timeout=1200)
                            (a.output / (task['name'] + '.stdout')).write_text(result.stdout)
                            (a.output / (task['name'] + '.stderr')).write_text(result.stderr)
                            atomic_write(a.output / (task['name'] + '-status.json'), {'exit_code': result.returncode,
                                         'passed': result.returncode == 0, 'validation_kind': 'real_model_cookbook_cli'})
                except Exception as exc:
                    row['error'] = repr(exc)
                finally:
                    stop.set()
                    monitor.join(timeout=1)
                    row.update(memory)
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            report['records'].append(row)
            atomic_write(report_path, report)
            print('DONE', backend['name'], modality, row['passed'], flush=True)
    # Equal tokens are a useful media/compiler parity check, not a quality claim.
    report['matching_input_tokens'] = {kind: len({r.get('input_tokens') for r in report['records'] if r['modality'] == kind and r['passed']}) == 1 for kind in fixtures}
    report['completed'] = True
    report['passed'] = all(r['passed'] for r in report['records']) and all(report['matching_input_tokens'].values())
    atomic_write(report_path, report)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
