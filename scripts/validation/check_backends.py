"""Fixed real-service protocol checks. A completed JSON report is required to pass.

No model downloads. Run separately from offline adapter tests. Preserve reports
even when a backend rejects a supported request or violates a numerical gate.
"""
import argparse
import base64
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('contract_protocol', ROOT / 'valen/decisions.py')
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)
FIXTURE = ROOT / 'tests/fixtures/backend-contract.json'


def prepared_cases(image=None, video=None):
    data = json.loads(FIXTURE.read_text())
    payloads = {}
    for name, path, kind in [('IMAGE', image, 'image/png'), ('VIDEO', video, 'video/mp4')]:
        if path:
            payloads['$' + name + '_DATA'] = 'data:' + kind + ';base64,' + base64.b64encode(Path(path).read_bytes()).decode()
    cases = []
    for case in data['cases']:
        raw = json.dumps(case)
        if any('$' + name + '_DATA' in raw and '$' + name + '_DATA' not in payloads for name in ['IMAGE', 'VIDEO']):
            continue
        for placeholder, value in payloads.items():
            raw = raw.replace(placeholder, value)
        case = json.loads(raw)
        cases.append((case['id'], {'input': case['input'], 'questions': case['questions'], 'temperature': data['temperature']}))
    return cases, data['invalid']


def post(base, body, route='decisions'):
    started = time.perf_counter()
    req = urllib.request.Request(base.rstrip('/') + '/' + route, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            result = json.load(response)
            inference = response.headers.get('X-Valen-Inference-Seconds')
            return response.status, result, time.perf_counter() - started, float(inference) if inference else None
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read()), time.perf_counter() - started, None


def validate(body, response):
    answers = response['answers']
    assert [a['name'] for a in answers] == [q['name'] for q in body['questions']], 'answer order'
    assert response['usage']['output_tokens'] == 0, 'Valen must not generate answer tokens'
    for q, a in zip(body['questions'], answers):
        assert a['type'] == q['type']
        if q['type'] == 'predicate':
            assert math.isfinite(a['probability']) and 0 <= a['probability'] <= 1
            continue
        probabilities = a['probabilities']
        key, source = ('value', 'choices') if q['type'] == 'choice' else ('label', 'levels')
        assert [(type(p[key]), p[key]) for p in probabilities] == [(type(c[key]), c[key]) for c in q[source]], 'typed candidate order'
        ps = [p['probability'] for p in probabilities]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in ps) and abs(sum(ps) - 1) < 1e-5
        if q['type'] == 'choice':
            expected = q['choices'][max(range(len(ps)), key=ps.__getitem__)]['value']
            assert type(a['choice']) is type(expected) and a['choice'] == expected
        else:
            assert [p['value'] for p in probabilities] == list(range(len(ps)))
            assert abs(a['score'] - sum(i * v for i, v in enumerate(ps))) < 1e-5


def probabilities(response):
    return [p for a in response['answers'] for p in
            ([a['probability']] if a['type'] == 'predicate' else [v['probability'] for v in a['probabilities']])]


def check(base, image=None, video=None):
    cases, invalid = prepared_cases(image, video)
    records = []
    for name, original in cases:
        for temperature, reverse in [(1.0, False), (0.5, False), (1.0, True)]:
            body = deepcopy(original)
            body['temperature'] = temperature
            if reverse:
                for q in body['questions']:
                    if q['type'] == 'choice':
                        q['choices'].reverse()
            record = {'id': name, 'temperature': temperature, 'reversed_choices': reverse, 'passed': False}
            try:
                status, public, *_ = post(base, body)
                assert status == 200, public
                validate(body, public)
                internal = PROTOCOL.to_internal(body, allow_video=True)
                internal['temperature'] = temperature
                status, legacy, *_ = post(base, internal, 'systemone')
                assert status == 200, legacy
                mapped = PROTOCOL.to_public(body, legacy)
                validate(body, mapped)
                delta = max(abs(x - y) for x, y in zip(probabilities(public), probabilities(mapped)))
                assert delta < 2e-5, ('same-backend protocol probability drift', delta)
                assert public['usage'] == legacy['usage'], 'protocol token usage drift'
                record.update(passed=True, protocol_max_probability_delta=delta, response=public)
            except Exception as exc:
                record['error'] = str(exc)
            records.append(record)
    for case in invalid:
        record = {'id': case['id'], 'passed': False}
        try:
            status, response, *_ = post(base, case['body'])
            record.update(status=status, response=response, passed=400 <= status < 500)
        except Exception as exc:
            record['error'] = str(exc)
        records.append(record)
    return {'validation_kind': 'real_model_http', 'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            'scope': [name for name, _ in cases], 'records': records,
            'passed': all(r['passed'] for r in records)}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--image')
    p.add_argument('--video')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = check(a.base_url, a.image, a.video)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
