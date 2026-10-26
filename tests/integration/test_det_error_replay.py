"""Only uniquely covered, original wrong training records may enter RLCD."""
import copy
import json

import pytest

from scripts.train.replay_det_errors import balanced_errors,inference_record,select_errors,score_training_errors


def records(count=60):
    return [{'id':str(i),'group_id':'image:'+str(i//3),'request':{'questions':{'q':{'instructions':'target'}}},
             'targets':{'q':{'boxes':[[.1,.1,.9,.9]],'labels':[0]}},
             'meta':{'source':['refcoco','refcoco+','reasonseg'][i%3],
                     'allow_empty_target':True,'private_ground_truth':'must not reach model'}} for i in range(count)]


def test_error_subset_is_unique_balanced_original_and_repeatable():
    original=records()
    selected=balanced_errors(original,cap=12)
    assert selected==balanced_errors(original,cap=12)
    assert len(selected)==len({r['id'] for r in selected})==12
    assert len({r['group_id'] for r in selected})==12
    assert {source:sum(r['meta']['source']==source for r in selected)
            for source in ['refcoco','refcoco+','reasonseg']}=={'refcoco':4,'refcoco+':4,'reasonseg':4}
    assert all(row==original[int(row['id'])] for row in selected)


def test_all_small_error_sets_are_used_without_duplication_and_coverage_is_mandatory(tmp_path):
    original=records(9)
    predictions=[{'id':r['id'],'correct':i>=2} for i,r in enumerate(original)]
    manifest=select_errors(original,predictions,'full-identity','sealed-sha',tmp_path)
    assert manifest['errors']==manifest['selected']==2
    assert len((tmp_path/'train.jsonl').read_text().splitlines())==2
    with pytest.raises(ValueError,match='coverage'):
        select_errors(original,predictions[:-1],'full-identity','sealed-sha',tmp_path)
    duplicate=copy.deepcopy(predictions);duplicate[-1]['id']=duplicate[0]['id']
    with pytest.raises(ValueError,match='IDs'):
        select_errors(original,duplicate,'full-identity','sealed-sha',tmp_path)


def test_inference_receives_no_targets_or_ground_truth_metadata():
    original=records(1)[0]
    request=inference_record(original)
    assert 'targets' not in request
    assert request['meta']=={'referring':True,'allow_empty_target':True}
    assert request['request']==original['request']
    request['request']['questions']['q']['instructions']='mutated'
    assert original['request']['questions']['q']['instructions']=='target'


def test_post_rl_scores_original_and_sealed_hard_subset_independently():
    original=records(9)
    predictions=[{'id':r['id'],'correct':i%2==0} for i,r in enumerate(original)]
    result=score_training_errors(original,predictions,[original[1],original[2]],'original-sha','rlcd-model')
    assert result['errors']==4 and result['selected_errors']==1 and result['selected_error_rate']==.5
    assert result['original_training_sha256']=='original-sha'
    with pytest.raises(ValueError,match='exactly once'):
        score_training_errors(original,predictions[:-1],original[:2],'original-sha','rlcd-model')
