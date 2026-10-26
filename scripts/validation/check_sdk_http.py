"""Real-service Python SDK regression using the shared protocol fixture."""
import argparse
import json
from pathlib import Path

from check_backends import PROTOCOL, prepared_cases, probabilities, validate


def main():
    from valen_sdk import Valen, ValenAPIError, __version__
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--image')
    p.add_argument('--video')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cases, invalid = prepared_cases(args.image, args.video)
    records = []
    with Valen.remote(args.base_url, max_retries=0) as client:
        for name, body in cases:
            record = {'id': name, 'passed': False}
            try:
                public = client.decisions(**body).model_dump()
                validate(body, public)
                internal = PROTOCOL.to_internal(body, allow_video=True)
                legacy = client.system_one(internal['state'], internal['questions'], temperature=body['temperature']).model_dump()
                mapped = PROTOCOL.to_public(body, legacy)
                validate(body, mapped)
                delta = max(abs(x-y) for x,y in zip(probabilities(public), probabilities(mapped)))
                assert delta < 2e-5
                record.update(passed=True, protocol_max_probability_delta=delta, response=public)
            except Exception as exc:
                record['error'] = str(exc)
            records.append(record)
        for case in invalid:
            record = {'id': case['id'], 'passed': False}
            try:
                client.decisions(**case['body'])
            except ValenAPIError as exc:
                record.update(status=exc.status_code, error=str(exc), passed=400 <= (exc.status_code or 0) < 500)
            except Exception as exc:
                record['error'] = repr(exc)
            records.append(record)
    report = {'passed': all(r['passed'] for r in records), 'validation_kind': 'real_model_python_sdk_http',
              'sdk_version': __version__, 'records': records}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
