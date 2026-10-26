"""Prepare complete UI-Vision, OSWorld-G and MMBench-GUI L2 grounding splits."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import subprocess
import zipfile

from scripts.data.prepare_gui_desktop import image_features, NearIndex
from scripts.data.prepare_spatial import digest_file, make_record, record_image, save_dataset, write_json
from scripts.data.prepare_spatial_gui import store_image


REVISIONS = {
    'uivision': 'ServiceNow/ui-vision@766c66aeffef16608d4916525902d9fb2598d7ce',
    'osworld': 'xlang-ai/OSWorld-G@daa6bd8e0e629f0917ad2984df930bf0bd967540',
    'mmbench': 'OpenGVLab/MMBench-GUI@e27757e3910e0d5995b811f916b509b4e34a4690',
}


def git_blob(path):
    data=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def provenance(source,kind):
    repo,revision=REVISIONS[kind].split('@')
    if kind=='uivision':
        inventory=json.loads((source/'protocol-v2-inventory.json').read_text())
        if inventory['repo']!=repo or inventory['revision']!=revision:
            raise ValueError('Unexpected UI-Vision source version')
        entries={entry['path']:entry for entry in inventory['tree'] if entry['type']=='file'}
        return entries
    if kind=='mmbench':
        metadata=json.loads((source/'source-metadata.json').read_text())
        if metadata['sha']!=revision or metadata['id']!=repo:
            raise ValueError('Unexpected MMBench-GUI source version')
        entries={entry['rfilename']:entry for entry in metadata['siblings']}
        path=source/'L2_annotations.json'
        if git_blob(path)!=entries[path.name]['blobId']:
            raise ValueError('MMBench-GUI annotations differ from the pinned source')
        if digest_file(source/'MMBench-GUI-OfflineImages.zip')!=entries['MMBench-GUI-OfflineImages.zip']['lfs']['sha256']:
            raise ValueError('MMBench-GUI archive differs from the pinned source')
        return entries
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if head!=revision:
        raise ValueError('Unexpected OSWorld-G source commit')
    entries={}
    for line in subprocess.check_output(['git','-C',str(source),'ls-tree','-r','HEAD'],text=True).splitlines():
        identity,name=line.split('\t',1)
        entries[name]=dict(oid=identity.split()[2])
    return entries


def verify_source_file(path,source,entries):
    entry=entries[str(path.relative_to(source))]
    if entry.get('lfs'):
        expected=entry['lfs'].get('oid',entry['lfs'].get('sha256'))
        valid=digest_file(path)==expected
    else:
        valid=git_blob(path)==entry.get('oid',entry.get('blobId'))
    if not valid:
        raise ValueError('Source file differs from the pinned official version: '+path.name)


def iter_samples(source, kind):
    if kind == 'uivision':
        for setting, count in [('basic', 1772), ('functional', 1772), ('spatial', 1935)]:
            path = source / f'annotations/element_grounding/element_grounding_{setting}.json'
            samples = json.loads(path.read_text())
            if len(samples) != count:
                raise ValueError('Incomplete UI-Vision split')
            for index, row in enumerate(samples):
                yield f'uivision_{setting}', index, row, source / 'images' / row['image_path'], path
    elif kind == 'osworld':
        path = source / 'benchmark/OSWorld-G_refined.json'
        samples = json.loads(path.read_text())
        if len(samples) != 564:
            raise ValueError('Incomplete OSWorld-G refined split')
        for index, row in enumerate(samples):
            yield 'osworld_g_refined', index, row, source / 'benchmark/images' / row['image_path'], path
    else:
        path = source / 'L2_annotations.json'
        samples = json.loads(path.read_text())
        if not isinstance(samples, list) or len(samples) != 3594:
            raise ValueError('MMBench-GUI L2 must contain all 3,594 annotations')
        with zipfile.ZipFile(source / 'MMBench-GUI-OfflineImages.zip') as archive:
            for index, row in enumerate(samples):
                relative = Path(row['platform']) / row['image_path']
                names = [str(Path(prefix) / relative) for prefix in
                         ('offline_images', 'MMBench-GUI-OfflineImages/offline_images')]
                matched = [name for name in names if name in archive.namelist()]
                if len(matched) != 1:
                    raise ValueError('Missing or ambiguous original MMBench-GUI screenshot')
                yield 'mmbench_gui_l2', index, row, archive.read(matched[0]), path


def convert(row, kind, size):
    width, height = size
    if tuple(row['image_size']) != tuple(size):
        raise ValueError('Original screenshot dimensions differ from the annotation')
    metadata = dict(official_sample=row, platform=row.get('platform', 'desktop'),
                    ui_type=row.get('element_type', row.get('data_type', 'unknown')))
    if kind == 'uivision':
        instruction = row['prompt_to_evaluate']
        x1, y1, x2, y2 = row['bbox']
        box = [min(x1,x2)/width, min(y1,y2)/height, max(x1,x2)/width, max(y1,y2)/height]
        metadata.update(application=row['platform'], target_type='bbox', official_box=box)
    elif kind == 'mmbench':
        instruction = row['instruction']
        box = list(row['bbox'])
        metadata.update(target_type='bbox', official_box=box, group=row['grounding_type'],
                        application=row.get('app_name','unknown'))
    else:
        instruction = row['instruction']
        target_type = row['box_type']
        coordinates = row['box_coordinates']
        metadata.update(target_type=target_type, gui_types=row['GUI_types'])
        if target_type == 'refusal':
            return instruction, [], metadata
        if target_type == 'bbox':
            x, y, w, h = coordinates
            box = [x/width, y/height, (x+w)/width, (y+h)/height]
            metadata['official_box'] = box
        elif target_type == 'polygon':
            if len(coordinates) < 6 or len(coordinates) % 2:
                raise ValueError('Invalid official polygon')
            points = [[x/width,y/height] for x,y in zip(coordinates[::2],coordinates[1::2])]
            metadata['official_polygon'] = points
            box = [min(p[0] for p in points),min(p[1] for p in points),
                   max(p[0] for p in points),max(p[1] for p in points)]
        else:
            raise ValueError('Unknown official target type')
    # Compiler labels require valid normalized boxes; scoring uses the original
    # box/polygon above, including official coordinates outside visible bounds.
    clipped = [max(0.,box[0]),max(0.,box[1]),min(1.,box[2]),min(1.,box[3])]
    if not clipped[0] < clipped[2] or not clipped[1] < clipped[3]:
        # UI-Vision Spatial includes one original zero-area point annotation.
        # Keep its exact official scoring geometry and denominator; do not
        # invent an enlarged target to satisfy the compiler's box schema.
        metadata['official_degenerate_or_outside_box']=True
        return instruction, [], metadata
    return instruction, [clipped], metadata


def prepare(source, kind, output):
    records, hashes, annotation_hashes, image_cache = {}, {}, {}, {}
    entries=provenance(source,kind)
    if kind=='osworld':
        buckets=source/'benchmark/buckets.json'
        verify_source_file(buckets,source,entries)
        write_json(output/'manifests/osworld-buckets.json',json.loads(buckets.read_text()))
    for name, index, row, image, annotation in iter_samples(source,kind):
        key = str(image) if isinstance(image,Path) else row['platform']+'/'+row['image_path']
        if key not in image_cache:
            if isinstance(image,Path):verify_source_file(image,source,entries)
            data = image.read_bytes() if isinstance(image,Path) else image
            image_cache[key] = store_image(output,data,kind)
        relative, digest, size = image_cache[key]
        hashes[relative] = digest
        annotation_name = str(annotation.relative_to(source))
        if annotation_name not in annotation_hashes:
            verify_source_file(annotation,source,entries)
            annotation_hashes[annotation_name] = digest_file(annotation)
        instruction, boxes, metadata = convert(row,kind,size)
        records.setdefault(name,[]).append(make_record('gui',f'{name}:{index}',digest,relative,
            instruction,['target'],boxes,[0]*len(boxes),source=kind,split='test',
            width=size[0],height=size[1],**metadata))
    manifests = {}
    for name, rows in records.items():
        manifests[name] = save_dataset(output,name,rows,[REVISIONS[kind]],hashes,
            original_annotation_sha256=annotation_hashes,complete_official_split=True,
            license='mit' if kind=='uivision' else 'apache-2.0',
            target_types=dict(Counter(r['meta']['target_type'] for r in rows)))
    return manifests


def audit_overlap(output, training_root, manifests, workers):
    seal_path = training_root / 'manifests/ready.json'
    seal_hash = digest_file(seal_path)
    train = json.loads((training_root/'manifests/train.json').read_text())
    seal = json.loads(seal_path.read_text())
    if digest_file(training_root/'processed/train.jsonl') != seal['records_sha256']:
        raise ValueError('Training corpus differs from its seal')
    media = sorted(train['media_sha256'])
    index = NearIndex()
    def feature(relative):
        result = image_features((training_root/relative).read_bytes())
        if result['raw'] != train['media_sha256'][relative]:
            raise ValueError('Training screenshot hash changed')
        return result
    with ThreadPoolExecutor(workers) as pool:
        for number, result in enumerate(pool.map(feature,media)):
            index.add(result)
            if number%2000==0: print(json.dumps(dict(phase='training_overlap_index',images=number)),flush=True)
    evaluation_media = sorted({p for m in manifests.values() for p in m['media_sha256']})
    overlaps = {}
    for relative in evaluation_media:
        f = image_features((output/relative).read_bytes())
        matches = index.matches(f)
        if matches:
            overlaps[relative] = [dict(training_image=media[i],
                match='file' if f['raw']==index.features[i]['raw'] else
                      'rgb' if f['pixels']==index.features[i]['pixels'] else 'near') for i in matches]
    if digest_file(seal_path) != seal_hash:
        raise ValueError('Training version changed during the overlap audit')
    counts = {name:sum(p in overlaps for p in manifest['media_sha256']) for name,manifest in manifests.items()}
    return dict(training_records_sha256=seal['records_sha256'],training_seal_sha256=seal_hash,
        training_images=len(media),evaluation_images=len(evaluation_media),
        overlap_images=counts,overlaps=overlaps,
        method='file SHA256, decoded RGB SHA256, dhash128 distance<=8 and thumbnail checks',
        full_splits_preserved=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in REVISIONS:
        parser.add_argument('--'+name+'-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--training-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    output=args.output_root
    if (output/'manifests/ready.json').exists():
        raise ValueError('Use a new output directory rather than overwriting a sealed benchmark')
    manifests={}
    for kind in REVISIONS:
        manifests.update(prepare(getattr(args,kind+'_root'),kind,output))
    overlap=audit_overlap(output,args.training_root,manifests,args.workers)
    write_json(output/'manifests/training-overlap.json',overlap)
    write_json(output/'manifests/ready.json',dict(validated=True,benchmarks={
        name:dict(samples=m['samples'],records_sha256=m['records_sha256']) for name,m in manifests.items()},
        osworld_buckets_sha256=digest_file(output/'manifests/osworld-buckets.json'),
        training_overlap_manifest_sha256=digest_file(output/'manifests/training-overlap.json'),
        official_revisions=REVISIONS,training_records_sha256=overlap['training_records_sha256']))


if __name__=='__main__':
    main()
