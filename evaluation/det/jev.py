"""Evaluate Jev's native Box/Noul API on sealed Det benchmarks."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shlex
import time
import uuid

import httpx
from PIL import Image


BENCHMARKS = tuple(f'{source}_{split}' for source in ('refcoco', 'refcocoplus')
                   for split in ('val', 'testA', 'testB')) + ('refcocog_val', 'refcocog_test',
                   'reasonseg_val', 'reasonseg_test', 'groundingme_test')
ENDPOINT = 'https://api.typesafe.ai/v1/systemone'


class InvalidBox(ValueError):
    """A delivered prediction has invalid coordinates; score it as a model failure."""


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_key(name, env_file=None):
    value = None
    if env_file:
        for line in Path(env_file).read_text().splitlines():
            key, separator, raw = line.strip().removeprefix('export ').partition('=')
            if separator and key.strip() == name:
                parts = shlex.split(raw, comments=True)
                if len(parts) == 1:
                    value = parts[0]
    value = value or os.environ.get(name)
    if not value or not value.strip():
        raise ValueError('Missing API credential: ' + name)
    return value.strip()


def image_asset(path, max_pixels, geometry=None):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    with Image.open(path) as source:
        image = source.convert('RGB')
        height, width = smart_resize(image.height, image.width, factor=32,
                                     min_pixels=4096, max_pixels=max_pixels)
        # Respect the service's documented 280..1568 side limits without changing
        # aspect ratio. Letterbox narrow images and invert that transform on output.
        low, high = 0., min(1., 1568 / max(width, height))
        if max(280, int(width * high)) * max(280, int(height * high)) <= max_pixels:
            low = high
        else:
            for _ in range(40):
                scale = (low + high) / 2
                if max(280, int(width * scale)) * max(280, int(height * scale)) <= max_pixels:
                    low = scale
                else:
                    high = scale
        width, height = max(1, int(width * low)), max(1, int(height * low))
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        canvas_width, canvas_height = max(280, width), max(280, height)
        left, top = (canvas_width - width) // 2, (canvas_height - height) // 2
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        canvas.paste(image, (left, top))
        if geometry is not None:
            geometry.update(content_width=width, content_height=height,
                            canvas_width=canvas_width, canvas_height=canvas_height,
                            left=left, top=top)
        output = io.BytesIO()
        canvas.save(output, format='PNG')
    return {'type': 'image', 'format': 'png',
            'data_base64': base64.b64encode(output.getvalue()).decode('ascii')}


def request_body(record, asset, model, allow_empty=False):
    # Only the image and original instruction reach the service; no labels or metadata.
    reference = uuid.uuid4().hex
    instruction = record['request']['questions']['target']['instructions']
    body = {'model': model, 'state': {'photo': reference}, 'assets': {reference: asset},
            'questions': {'target': {'type': 'box', 'instructions': 'In `photo`: ' + instruction}}}
    if allow_empty:
        # The same extra question is asked for every example in these benchmarks.
        body['questions']['present'] = {'type': 'noul', 'instructions':
            'Does `photo` contain a visible target satisfying all requirements in this instruction?\n' + instruction}
    return body


def restore_box(box, geometry):
    if not geometry:
        return box
    result = [(box[index] * geometry['canvas_' + axis] - geometry[offset]) / geometry['content_' + axis]
              for index, axis, offset in ((0, 'width', 'left'), (1, 'height', 'top'),
                                          (2, 'width', 'left'), (3, 'height', 'top'))]
    result = [max(0., min(1., v)) for v in result]
    return result if result[0] < result[2] and result[1] < result[3] else []


def parse_presence(response):
    answer = response['answers']['present']
    value = answer.get('noul')
    if answer.get('type') != 'noul' or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid native Noul presence answer')
    return value


def post_with_retry(client, body, retries):
    for attempt in range(retries + 1):
        try:
            response = client.post(ENDPOINT, json=body)
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == retries:
                return response
            delay = response.headers.get('Retry-After', '')
            delay = float(delay) if delay.replace('.', '', 1).isdigit() else 2 ** attempt
        except httpx.TransportError:
            if attempt == retries:
                raise
            delay = 2 ** attempt
        time.sleep(min(30, max(1, delay)))


def parse_box(response, model):
    if response.get('model') != model:
        raise ValueError('The service returned a different model version')
    answer = response['answers']['target']
    if answer.get('type') != 'box':
        raise ValueError('The service did not return a native Box answer')
    box = [answer['box'][name] for name in ('left', 'top', 'right', 'bottom')]
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
            or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
        raise InvalidBox('Invalid normalized xyxy Box answer')
    confidence = answer.get('confidence')
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        raise ValueError('Invalid Box confidence')
    return box, confidence


def iou(first, second):
    if not first or not second:
        return float(not first and not second)
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return intersection / (area(first) + area(second) - intersection)


def evaluate(args):
    key = load_key(args.api_key_env, args.env_file)
    root, output = args.data_root.resolve(), args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    datasets, identities = {}, {}
    for name in args.benchmarks:
        base = root / ('reasonseg' if name.startswith('reasonseg_') else 'groundingme' if name.startswith('groundingme_') else '')
        path = base / 'processed' / (name + '.jsonl')
        manifest = json.loads((base / 'manifests' / (name + '.json')).read_text())
        if name == 'groundingme_test':
            from evaluation.det.groundingme import DATASET_REVISION, SCORER_REVISION
            preflight = json.loads((base / 'manifests/preflight.json').read_text())
            if (manifest.get('samples') != 1005 or manifest.get('source_revision') != DATASET_REVISION
                    or manifest.get('scorer_revision') != SCORER_REVISION or not preflight.get('validated')
                    or preflight.get('baseline_training_sha256_overlap') != 0):
                raise ValueError('GroundingME requires the complete pinned and isolated test split')
        if digest(path) != manifest['records_sha256']:
            raise ValueError('The benchmark records changed: ' + name)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != manifest['samples'] or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Benchmark count or sample IDs differ: ' + name)
        allows_empty = name.startswith(('reasonseg_', 'groundingme_'))
        if any(len(row['targets']['target']['boxes']) not in ((0, 1) if allows_empty else (1,)) for row in rows):
            raise ValueError('Localization requires one reference box or a supported empty target')
        random.Random(args.seed).shuffle(rows)
        selected = rows[:args.limit_per_benchmark] if args.limit_per_benchmark else rows
        identities[name] = dict(records_sha256=manifest['records_sha256'],
            manifest_sha256=digest(base / 'manifests' / (name + '.json')),
            full_samples=len(rows), selected_samples=len(selected))
        datasets[name] = (selected, manifest, base)
    threshold = getattr(args, 'presence_threshold', .5)
    config = dict(format='valen_jev_det_v2', model=args.model, endpoint=ENDPOINT, code_sha256=digest(__file__),
        max_pixels=args.max_pixels, resize='Qwen grid/bounded 1568; centered white padding to side280; inverse affine xyxy',
        provider_internal_preprocessing='unspecified', limit_per_benchmark=args.limit_per_benchmark,
        seed=args.seed, benchmarks=identities, question_type='box', confidence_filter=False,
        absence_rule=dict(benchmarks=['reasonseg_val', 'reasonseg_test', 'groundingme_test'],
                          primitive='noul', present_at_or_above=threshold, threshold_fitted=False),
        ground_truth_excluded=True, score='normalized xyxy IoU >= 0.5')
    config_path = output / 'request-config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Use a new output directory for changed data or request settings')
    write_json(config_path, config)
    metrics = {}
    completed = 0
    stored_count = sum(1 for path in output.glob('*.jsonl') for line in path.open() if line.strip())
    with httpx.Client(timeout=httpx.Timeout(args.timeout, connect=min(10, args.timeout)),
            follow_redirects=False, headers={'Authorization': 'Bearer ' + key,
                                            'TypeSafe-Danger-Zone-Images': 'true'}) as client:
        try:
            for name, (rows, manifest, base) in datasets.items():
                allows_empty = name.startswith(('reasonseg_', 'groundingme_'))
                predictions = output / (name + '.jsonl')
                cached = [json.loads(line) for line in predictions.read_text().splitlines()] if predictions.exists() else []
                by_id = {row['id']: row for row in cached}
                ids = {row['id'] for row in rows}
                if len(by_id) != len(cached) or not set(by_id) <= ids:
                    raise ValueError('Duplicate or unexpected cached prediction IDs')
                with predictions.open('a') as stream:
                    for record in rows:
                        if record['id'] in by_id:
                            continue
                        relative = record['request']['state']['messages'][0]['content'][0]['image_url']['url']
                        image = (base / relative).resolve()
                        if not image.is_relative_to(base) or digest(image) != manifest['media_sha256'][relative]:
                            raise ValueError('Benchmark image identity changed')
                        started = time.monotonic()
                        geometry = {}
                        asset = image_asset(image, args.max_pixels, geometry)
                        response = post_with_retry(client, request_body(record, asset, args.model, allows_empty),
                                                   getattr(args, 'max_retries', 0))
                        if response.status_code != 200:
                            raise RuntimeError('Jev API HTTP ' + str(response.status_code))
                        raw = response.json()
                        presence = parse_presence(raw) if allows_empty else 1.
                        valid_output = True
                        try:
                            raw_box, confidence = parse_box(raw, args.model)
                        except InvalidBox:
                            raw_box, confidence = [], None
                            valid_output = allows_empty and presence < threshold
                        box = restore_box(raw_box, geometry) if presence >= threshold and raw_box else []
                        if presence >= threshold and not box:
                            valid_output = False
                        truths = record['targets']['target']['boxes']
                        truth = truths[0] if truths else []
                        row = dict(id=record['id'], box=box, raw_box=raw_box, geometry=geometry,
                                   presence=presence, confidence=confidence, valid_output=valid_output,
                                   iou=iou(box, truth) if valid_output else 0.,
                                   model=raw['model'], usage=raw.get('usage', {}),
                                   elapsed_seconds=time.monotonic() - started)
                        if name == 'groundingme_test':
                            row.update(subtask_l1=record['meta']['subtask_l1'], subtask_l2=record['meta']['subtask_l2'])
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        stream.flush()
                        by_id[row['id']] = row
                        completed += 1
                        stored_count += 1
                        if completed % 10 == 0:
                            print(json.dumps(dict(benchmark=name, new_predictions=completed)), flush=True)
                        if completed % 25 == 0:
                            write_json(output / 'status.json', dict(status='running', benchmark=name,
                                predictions_completed=stored_count,
                                time_utc=datetime.now(timezone.utc).isoformat()))
                values = list(by_id.values())
                metrics[name] = dict(samples=len(values), full_samples=manifest['samples'],
                    complete_split=len(values) == manifest['samples'], data_sha256=manifest['records_sha256'],
                    accuracy=sum(row['iou'] >= .5 for row in values) / len(values),
                    mean_box_iou=sum(row['iou'] for row in values) / len(values),
                    input_tokens=sum(row.get('usage', {}).get('input_tokens', 0) for row in values),
                    output_tokens=sum(row.get('usage', {}).get('output_tokens', 0) for row in values))
                if name == 'groundingme_test':
                    from evaluation.det.groundingme import aggregate
                    metrics[name].update(aggregate(values))
                write_json(output / 'partial-metrics.json', dict(model=args.model, partial=True, benchmarks=metrics))
            write_json(output / 'metrics.json', dict(model=args.model, protocol=config['format'],
                complete_full_benchmarks=all(value['complete_split'] for value in metrics.values()), benchmarks=metrics))
            write_json(output / 'status.json', dict(status='completed', predictions_completed=sum(v['samples'] for v in metrics.values()),
                full_evaluation=all(value['complete_split'] for value in metrics.values())))
        except Exception as error:
            # No transport/authentication failure is reported as a zero model score.
            write_json(output / 'status.json', dict(status='incomplete', new_predictions=completed,
                error_type=type(error).__name__, error=str(error), time_utc=datetime.now(timezone.utc).isoformat()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='jev-1.14.0')
    parser.add_argument('--api-key-env', default='JEV_API_KEY')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument('--limit-per-benchmark', type=int, default=4, help='Pilot samples per split; 0 runs each complete split')
    parser.add_argument('--max-pixels', type=int, default=262144)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-retries', type=int, default=4)
    parser.add_argument('--presence-threshold', type=float, default=.5,
                        help='Fixed native Noul threshold for every ReasonSeg/GroundingME record; never test-fitted')
    args = parser.parse_args()
    if args.limit_per_benchmark < 0 or args.max_pixels < 280**2 or args.timeout <= 0 or args.max_retries < 0 or not 0 <= args.presence_threshold <= 1:
        parser.error('Invalid sample limit, image budget or timeout')
    from filelock import FileLock
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output_dir / '.evaluation.lock'), timeout=0):
        evaluate(args)


if __name__ == '__main__':
    main()
