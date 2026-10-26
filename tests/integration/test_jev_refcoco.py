"""Native Jev boxes, label isolation, API failures and resumable pilot scoring."""
import json
from types import SimpleNamespace

import httpx
import pytest

from evaluation.det import jev


def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    (root / 'processed').mkdir(parents=True)
    (root / 'manifests').mkdir()
    (root / 'image.png').write_bytes(b'image placeholder')
    rows = [dict(id=str(i), group_id='image', request=dict(state=dict(messages=[dict(role='user', content=[
        dict(type='image_url', image_url=dict(url='image.png'))])]), questions=dict(target=dict(type='det',
        instructions='the left person', categories=['target']))), targets=dict(target=dict(boxes=[[.1,.2,.4,.8]], labels=[0])))
        for i in range(2)]
    path = root / 'processed/refcoco_val.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (root / 'manifests/refcoco_val.json').write_text(json.dumps(dict(samples=2, records_sha256=jev.digest(path),
        media_sha256={'image.png':jev.digest(root/'image.png')})))
    monkeypatch.setenv('JEV_API_KEY', 'unit-test-credential')
    monkeypatch.setattr(jev, 'image_asset', lambda *args:{'type':'image','format':'png','data_base64':'aW1hZ2U='})
    args=SimpleNamespace(api_key_env='JEV_API_KEY',env_file=None,data_root=root,output_dir=tmp_path/'output',
        model='jev-1.14.0',benchmarks=['refcoco_val'],limit_per_benchmark=1,max_pixels=262144,timeout=5,seed=42)
    return args, rows


def test_request_contains_no_ground_truth_or_record_metadata():
    record=dict(request=dict(questions=dict(target=dict(instructions='the left person'))),
                targets={'confidential_gt':'do not send'},meta={'annotation':'also not sent'})
    body=jev.request_body(record,{'type':'image','format':'png','data_base64':'abc'},'jev-1.14.0')
    assert set(body)=={'model','state','assets','questions'}
    assert body['questions']['target']=={'type':'box','instructions':'In `photo`: the left person'}
    assert 'do not send' not in json.dumps(body) and 'also not sent' not in json.dumps(body)
    assert body['state']['photo'] in body['assets']


def test_native_edges_have_fixed_xyxy_order_and_pixels_are_rejected():
    answer=dict(model='jev-1.14.0',answers={'target':{'type':'box','box':dict(left=.1,top=.2,right=.4,bottom=.8),'confidence':.7}})
    box,confidence=jev.parse_box(answer,'jev-1.14.0')
    assert box==[.1,.2,.4,.8] and confidence==.7 and jev.iou(box,box)==1
    answer['answers']['target']['box']['right']=400
    with pytest.raises(ValueError,match='normalized'):
        jev.parse_box(answer,'jev-1.14.0')


@pytest.mark.parametrize('size', [(640,210), (2000,80), (80,2000), (640,491), (50,50)])
def test_service_padding_stays_in_budget_and_inverts_without_labels(tmp_path,size):
    import base64,io
    from PIL import Image
    path=tmp_path/'image.png';Image.new('RGB',size,'red').save(path)
    geometry={};asset=jev.image_asset(path,262144,geometry)
    with Image.open(io.BytesIO(base64.b64decode(asset['data_base64']))) as image:
        assert min(image.size)>=280 and max(image.size)<=1568
        assert image.width*image.height<=262144
    g=geometry
    box=[g['left']/g['canvas_width'],g['top']/g['canvas_height'],
         (g['left']+g['content_width'])/g['canvas_width'],(g['top']+g['content_height'])/g['canvas_height']]
    assert jev.restore_box(box,g)==pytest.approx([0,0,1,1])


def test_native_presence_uses_the_same_request_for_positive_and_empty_labels():
    positive=dict(request=dict(questions=dict(target=dict(instructions='the red car'))),targets=dict(target=dict(boxes=[[.1,.1,.2,.2]])))
    negative=dict(positive,targets=dict(target=dict(boxes=[])))
    first=jev.request_body(positive,{},'jev-1.14.0',True)
    second=jev.request_body(negative,{},'jev-1.14.0',True)
    assert first['questions']==second['questions']
    assert first['questions']['present']['type']=='noul'
    assert jev.parse_presence(dict(answers=dict(present=dict(type='noul',noul=.1))))==.1
    assert jev.iou([],[])==1 and jev.iou([],[.1,.1,.2,.2])==0


def test_transport_failure_never_becomes_a_zero_accuracy(tmp_path,monkeypatch):
    args,_=fixture(tmp_path,monkeypatch)
    class Offline:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,*args,**kwargs):raise httpx.ConnectError('offline')
    monkeypatch.setattr(jev.httpx,'Client',Offline)
    with pytest.raises(httpx.ConnectError):jev.evaluate(args)
    assert not (args.output_dir/'metrics.json').exists()
    status=json.loads((args.output_dir/'status.json').read_text())
    assert status['status']=='incomplete' and status['new_predictions']==0
    assert 'unit-test-credential' not in (args.output_dir/'request-config.json').read_text()
    assert 'unit-test-credential' not in (args.output_dir/'status.json').read_text()


def test_pilot_is_labeled_and_resume_does_not_repeat_paid_predictions(tmp_path,monkeypatch):
    args,_=fixture(tmp_path,monkeypatch);calls=[]
    class Online:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,url,**kwargs):
            assert url==jev.ENDPOINT
            calls.append(kwargs['json'])
            return httpx.Response(200,json=dict(model='jev-1.14.0',usage=dict(input_tokens=10,output_tokens=0),
                answers={'target':dict(type='box',box=dict(left=.1,top=.2,right=.4,bottom=.8),confidence=.7)}))
    monkeypatch.setattr(jev.httpx,'Client',Online)
    jev.evaluate(args);jev.evaluate(args)
    assert len(calls)==1
    report=json.loads((args.output_dir/'metrics.json').read_text())
    assert report['complete_full_benchmarks'] is False
    assert report['benchmarks']['refcoco_val']['accuracy']==1
    assert report['benchmarks']['refcoco_val']['samples']==1
    args.limit_per_benchmark=0
    with pytest.raises(ValueError,match='new output directory'):
        jev.evaluate(args)


def test_reasonseg_empty_targets_use_native_presence_and_keep_full_coverage(tmp_path,monkeypatch):
    args,rows=fixture(tmp_path,monkeypatch)
    rows[1]['targets']['target']['boxes']=[]
    rows[1]['targets']['target']['labels']=[]
    base=args.data_root/'reasonseg';(base/'processed').mkdir(parents=True);(base/'manifests').mkdir()
    (base/'image.png').write_bytes(b'image placeholder')
    path=base/'processed/reasonseg_val.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (base/'manifests/reasonseg_val.json').write_text(json.dumps(dict(samples=2,records_sha256=jev.digest(path),
        media_sha256={'image.png':jev.digest(base/'image.png')})))
    args.benchmarks=['reasonseg_val'];args.limit_per_benchmark=0
    calls=[]
    class Online:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,url,**kwargs):
            calls.append(kwargs['json'])
            return httpx.Response(200,json=dict(model='jev-1.14.0',answers=dict(
                target=dict(type='box',box=dict(left=.1,top=.2,right=.4,bottom=.8),confidence=.7),
                present=dict(type='noul',noul=0. if len(calls)==1 else 1.))))
    monkeypatch.setattr(jev.httpx,'Client',Online)
    jev.evaluate(args)
    assert len(calls)==2 and calls[0]['questions']==calls[1]['questions']
    report=json.loads((args.output_dir/'metrics.json').read_text())
    assert report['complete_full_benchmarks']
    assert report['benchmarks']['reasonseg_val']['samples']==2
    assert report['benchmarks']['reasonseg_val']['accuracy']==1
