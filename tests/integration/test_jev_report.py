"""Completed benchmark scores require exact ID coverage and matching source data."""
import copy
import json

import pytest

from evaluation.det.jev_report import comparison_rows,verify_predictions


def predictions():
    return [dict(id='a',model='jev-1.14.0',iou=1.,valid_output=True,elapsed_seconds=1.),
            dict(id='b',model='jev-1.14.0',iou=.3,valid_output=True,elapsed_seconds=2.)]


def test_complete_split_is_verified_from_the_actual_prediction_cache(tmp_path):
    path=tmp_path/'refcoco_val.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in predictions()))
    result=verify_predictions(path,{'a','b'},'jev-1.14.0',dict(samples=2,accuracy=.5,mean_box_iou=.65))
    assert result['all_expected_ids_verified'] and result['samples']==2
    assert result['request_latency_seconds']['mean']==1.5


@pytest.mark.parametrize('failure',['missing','duplicate','model','iou','invalid_correct','reported_score'])
def test_partial_wrong_model_or_wrong_scores_cannot_be_reported_as_complete(tmp_path,failure):
    rows=copy.deepcopy(predictions());metrics=dict(samples=2,accuracy=.5,mean_box_iou=.65)
    if failure=='missing':rows.pop()
    if failure=='duplicate':rows[1]['id']='a'
    if failure=='model':rows[0]['model']='different-model'
    if failure=='iou':rows[0]['iou']=float('nan')
    if failure=='invalid_correct':rows[0]['valid_output']=False
    if failure=='reported_score':metrics['accuracy']=1.
    path=tmp_path/'refcoco_val.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError):verify_predictions(path,{'a','b'},'jev-1.14.0',metrics)


def test_jev_comparison_requires_identical_records_and_keeps_incomplete_api_cells_empty():
    config=dict(benchmarks=dict(refcoco_val=dict(full_samples=2,records_sha256='same-data')))
    baseline=dict(benchmarks=dict(refcoco_val=dict(samples=2,data_sha256='same-data',accuracy=.8)))
    row=comparison_rows(config,{},dict(test=baseline))['refcoco_val']
    assert row['jev_accuracy'] is None and row['valen_minus_jev_pp']=={}
    result=comparison_rows(config,dict(refcoco_val=dict(accuracy=.5)),dict(test=baseline))
    assert result['refcoco_val']['valen_minus_jev_pp']['test']==pytest.approx(30)
    baseline['benchmarks']['refcoco_val']['data_sha256']='different'
    with pytest.raises(ValueError,match='different records'):comparison_rows(config,{},dict(test=baseline))


def test_cached_iou_is_recomputed_against_original_annotations(tmp_path):
    row=dict(id='a',model='jev-1.14.0',box=[.1,.1,.4,.4],iou=1.,valid_output=True,elapsed_seconds=1.)
    path=tmp_path/'refcoco_val.jsonl';path.write_text(json.dumps(row)+'\n')
    metrics=dict(samples=1,accuracy=1.,mean_box_iou=1.)
    correct={'a':[[.1,.1,.4,.4]]}
    assert verify_predictions(path,{'a'},'jev-1.14.0',metrics,truth_boxes=correct)['original_box_iou_recomputed']
    with pytest.raises(ValueError,match='original labeled box'):
        verify_predictions(path,{'a'},'jev-1.14.0',metrics,truth_boxes={'a':[[.6,.6,.9,.9]]})
