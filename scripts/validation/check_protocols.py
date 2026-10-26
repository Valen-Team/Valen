"""Fast synthetic adapter checks against one shared fixture; no inference."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'tests/fixtures/backend-contract.json'


def load(path, index):
    spec = importlib.util.spec_from_file_location('protocol_check_' + str(index), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_response(internal):
    answers = {}
    for name, question in internal['questions'].items():
        kind = question['type']
        if kind == 'noul':
            answers[name] = {'type': kind, 'noul': .75}
        else:
            keys = list(question['criteria']) if kind == 'choice' else [str(i) for i in range(len(question['criteria']))]
            total = sum(range(1, len(keys) + 1))
            ps = [(i + 1) / total for i in range(len(keys))]
            answers[name] = {'type': kind, 'probabilities': dict(zip(keys, ps)), 'confidence': .5}
            if kind == 'choice':
                answers[name]['choice'] = keys[-1]
            else:
                answers[name]['score'] = sum(i * v for i, v in enumerate(ps))
    return {'model': 'synthetic-not-a-model', 'answers': answers, 'usage': {'input_tokens': 42, 'output_tokens': 0}}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--python-sdk-dir', type=Path)
    parser.add_argument('--javascript-sdk-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    paths = [ROOT / 'valen/decisions.py', ROOT / 'serving/sglang/src/valen_sglang/decisions.py',
             ROOT / 'serving/vllm/src/valen_vllm/decisions.py']
    if args.python_sdk_dir:
        paths.append(args.python_sdk_dir / 'src/valen_sdk/protocol.py')
    modules = [load(path, index) for index, path in enumerate(paths)]
    data = json.loads(FIXTURE.read_text())
    payload = []
    for case in data['cases']:
        for temperature in [1.0, .5]:
            for reverse in [False, True]:
                body = {'input': deepcopy(case['input']), 'questions': deepcopy(case['questions']), 'temperature': temperature}
                if reverse:
                    for question in body['questions']:
                        if question['type'] == 'choice':
                            question['choices'].reverse()
                original = deepcopy(body)
                internal = modules[0].to_internal(body, allow_video=True)
                response = synthetic_response(internal)
                public = modules[0].to_public(body, response)
                for module in modules:
                    assert module.to_internal(body, allow_video=True) == internal, 'request mapping drift'
                    assert module.to_public(body, response) == public, 'response mapping drift'
                assert body == original, 'adapter mutated the input'
                assert [a['name'] for a in public['answers']] == [q['name'] for q in body['questions']]
                for question, answer in zip(body['questions'], public['answers']):
                    if question['type'] == 'choice':
                        expected = question['choices'][-1]['value']
                        assert type(answer['choice']) is type(expected) and answer['choice'] == expected
                payload.append({'body': body, 'internal': internal, 'response': response, 'public': public})
    for case in data['invalid']:
        for module in modules:
            try:
                module.to_internal(case['body'], allow_video=True)
            except ValueError:
                continue
            raise AssertionError('Invalid request accepted: ' + case['id'])
    if args.javascript_sdk_dir:
        process = subprocess.run(['node', str(ROOT / 'scripts/validation/check_protocols.mjs'),
                    str(args.javascript_sdk_dir.resolve()), str(FIXTURE)], input=json.dumps(payload),
                    text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError(process.stdout + process.stderr)
    report = {'passed': True, 'validation_kind': 'synthetic_adapter',
              'scope': 'request/response mapping, candidate type/order, two scalar temperatures, invalid inputs; no model execution',
              'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), 'valid_variants': len(payload),
              'invalid_cases': len(data['invalid']), 'python_adapters': len(modules),
              'javascript_adapter': bool(args.javascript_sdk_dir)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
