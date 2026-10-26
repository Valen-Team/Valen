"""Official grounding geometry and aggregates retain difficult and empty targets."""
import pytest

from evaluation.gui.additional import EXPECTED, aggregate, grounding_point, official_summary, results_ready, score
from scripts.data.prepare_gui_benchmarks import convert


def test_uivision_pixel_boxes_use_original_size_and_inclusive_boundaries():
    row=dict(image_size=[2000,1000],prompt_to_evaluate='Save',bbox=[1100,400,1000,450],
             platform='Editor',element_type='icon')
    instruction,boxes,metadata=convert(row,'uivision',(2000,1000))
    assert instruction=='Save' and boxes==[[.5,.4,.55,.45]]
    assert score([.5,.4],metadata) and score([.55,.45],metadata)
    assert not score([.56,.45],metadata)
    with pytest.raises(ValueError):convert(row,'uivision',(1000,500))


def test_osworld_concave_polygon_is_not_scored_as_its_bounding_rectangle():
    row=dict(image_size=[100,100],instruction='Select the L-shaped region',
             box_type='polygon',box_coordinates=[10,10,90,10,90,30,30,30,30,90,10,90],GUI_types=['Panel/Container'])
    _,boxes,metadata=convert(row,'osworld',(100,100))
    assert boxes==[[.1,.1,.9,.9]]
    assert score([.2,.8],metadata)
    assert not score([.8,.8],metadata)


def test_original_zero_area_annotation_is_retained_without_enlargement():
    row=dict(image_size=[100,100],prompt_to_evaluate='Closest element',
             bbox=[25,30,25,30],platform='Editor')
    _,boxes,metadata=convert(row,'uivision',(100,100))
    assert boxes==[] and metadata['official_box']==[.25,.3,.25,.3]
    assert score([.25,.3],metadata)
    assert not score([.250001,.3],metadata)


def test_osworld_refusal_requires_a_valid_explicit_negative_point():
    row=dict(image_size=[100,100],instruction='Locate a missing element',
             box_type='refusal',box_coordinates=[],GUI_types=['/'])
    _,boxes,metadata=convert(row,'osworld',(100,100))
    assert boxes==[] and score([-1,-1],metadata)
    for point in (None,[float('nan'),-1],[-1,.5],[.2,.4]):
        assert not score(point,metadata)


def test_only_a_categorical_null_action_can_count_as_model_refusal():
    from types import SimpleNamespace
    import torch
    question=SimpleNamespace(kind='gui',categories=['target'],metadata={},original_size=(100,100))
    def output(logits,box):
        return SimpleNamespace(class_logits=torch.tensor([logits],dtype=torch.float),
            boxes=torch.tensor([box],dtype=torch.float),mask_logits=None)
    positive=output([10.,-10.],[.2,.3,.6,.7])
    assert grounding_point(question,positive)==pytest.approx([.4,.5])
    assert grounding_point(question,output([-10.,10.],[.2,.3,.6,.7]))==[-1.,-1.]
    invalid=output([10.,-10.],[.6,.3,.2,.7])
    assert grounding_point(question,invalid) is None
    assert not score(grounding_point(question,invalid),dict(target_type='refusal'))
    assert grounding_point(question,output([float('nan'),1.],[.2,.3,.6,.7])) is None
    assert question.metadata=={}


def test_original_empty_instruction_is_opt_in_for_evaluation_and_compiled_verbatim(tmp_path):
    import copy,json
    from types import SimpleNamespace
    import torch
    from PIL import Image
    from scripts.data.prepare_spatial import make_record
    from valen.data.spatial import SpatialCompiler,read_spatial_jsonl,validate_spatial_record
    record=make_record('gui','blank','image','image.png','',['target'],[[.1,.1,.9,.9]],[0])
    path=tmp_path/'records.jsonl';path.write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError,match='nonempty'):validate_spatial_record(record)
    with pytest.raises(ValueError,match='nonempty'):read_spatial_jsonl(path)
    assert read_spatial_jsonl(path,allow_empty_instruction=True)==[record]
    invalid=copy.deepcopy(record);invalid['targets']['target']['boxes']=[[.9,.1,.1,.9]]
    with pytest.raises(ValueError,match='xyxy'):validate_spatial_record(invalid,allow_empty_instruction=True)
    segments=[]
    class Processor:
        tokenizer=SimpleNamespace(all_special_tokens=[],encode=lambda text,**kw:segments.append(text) or [1]*len(text))
        image_processor=SimpleNamespace(merge_size=1)
        def apply_chat_template(self,*a,**kw):return 'image'
        def __call__(self,**kw):return dict(input_ids=torch.ones((1,1),dtype=torch.long),image_grid_thw=torch.tensor([[1,2,2]]))
    Image.new('RGB',(32,32)).save(tmp_path/'image.png')
    compiler=SpatialCompiler(Processor(),tmp_path)
    with pytest.raises(ValueError,match='nonempty'):compiler.compile(record)
    compiler.allow_empty_instruction=True
    assert len(compiler.compile(record).questions)==1
    assert 'Instruction: \nCategories:' in ''.join(segments)


def test_full_split_scores_keep_missing_predictions_in_the_denominator():
    rows=[dict(correct=True,groups=dict(platform='os_linux',grounding_type='basic')),
          dict(correct=False,groups=dict(platform='os_linux',grounding_type='advanced')),
          dict(correct=False,groups=dict(platform='os_web',grounding_type='advanced'))]
    result=aggregate(rows)
    assert result['samples']==3 and result['accuracy']==1/3
    assert result['groups']['grounding_type/advanced']==dict(samples=2,accuracy=0)


def test_official_summary_averages_settings_instead_of_pooling_unequal_counts():
    benchmarks={f'uivision_{kind}':dict(accuracy=a) for kind,a in
                [('basic',1.),('functional',0.),('spatial',0.)]}
    benchmarks['mmbench_gui_l2']=dict(accuracy=.9,groups={
        'grounding_type/basic':dict(accuracy=1.),'grounding_type/advanced':dict(accuracy=0.)})
    benchmarks['osworld_g_refined']=dict(accuracy=.25)
    assert official_summary(benchmarks)==dict(uivision_element_average=1/3,
        mmbench_l2_average=.5,osworld_g_refined_accuracy=.25)


def test_full_result_gate_rejects_wrong_export_data_partial_coverage_and_wrong_aggregation(tmp_path):
    import copy,json
    from scripts.data.prepare_spatial import digest_file
    path=tmp_path/'export_manifest.json'
    path.write_text(json.dumps(dict(model_name='gui-test',training_data_sha256='training')))
    identity=digest_file(path)
    seal=dict(validated=True,training_records_sha256='training',benchmarks={
        name:dict(samples=count,records_sha256=name) for name,count in EXPECTED.items()})
    rows={name:dict(samples=count,data_sha256=name,accuracy=.5,
        groups={'platform/test':dict(samples=count,accuracy=.5)}) for name,count in EXPECTED.items()}
    rows['mmbench_gui_l2']['groups'].update({
        'grounding_type/basic':dict(samples=2000,accuracy=.4),
        'grounding_type/advanced':dict(samples=1594,accuracy=.6)})
    report=dict(model='gui-test',model_export_manifest_sha256=identity,complete_official_splits=True,
        training_overlap=dict(training_records_sha256='training'),benchmarks=rows,official_summary=official_summary(rows))
    marker=dict(model='gui-test',model_export_manifest_sha256=identity,samples=9637)
    assert results_ready(report,marker,seal,path)
    for changed in ('model','records','count','aggregation','nan'):
        bad=copy.deepcopy(report)
        if changed=='model':bad['model']='another-model'
        if changed=='records':bad['benchmarks']['uivision_basic']['data_sha256']='other-data'
        if changed=='count':bad['benchmarks']['uivision_basic']['samples']-=1
        if changed=='aggregation':bad['official_summary']['mmbench_l2_average']=.9
        if changed=='nan':bad['benchmarks']['uivision_basic']['accuracy']=float('nan')
        assert not results_ready(bad,marker,seal,path),changed
    path.write_text(path.read_text()+'\n')
    assert not results_ready(report,marker,seal,path)
