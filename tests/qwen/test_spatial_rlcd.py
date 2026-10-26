"""Joint action probabilities, reference KL and spatial policy gradients."""
import itertools
from types import SimpleNamespace

import pytest
import torch

from valen.modeling.qwen.spatial import SpatialOutput
from valen.training.spatial_rlcd_loss import SpatialActions, SpatialPolicy, SpatialRLCDObjective


def output(seed=0):
    generator=torch.Generator().manual_seed(seed)
    return SpatialOutput(torch.randn(2,2,generator=generator,requires_grad=True),
                         torch.randn(2,4,2,generator=generator,requires_grad=True))


def all_actions():
    coordinates=list(itertools.product(range(2), repeat=4))
    return SpatialActions(torch.tensor([False]+[True]*32),
                          torch.tensor([0]+[q for q in range(2) for _ in coordinates]),
                          torch.tensor([[0]*4]+coordinates+coordinates))


def test_factorized_kl_equals_enumerating_null_and_all_latent_boxes():
    first,second=SpatialPolicy(output(0),True),SpatialPolicy(output(1),True)
    actions=all_actions()
    left,right=first.log_prob(actions),second.log_prob(actions)
    assert left.exp().sum().item()==pytest.approx(1,abs=1e-6)
    exact=(left.exp()*(left-right)).sum()
    torch.testing.assert_close(first.kl(second),exact,atol=1e-6,rtol=1e-6)
    assert first.kl(first).item()==0
    assert first.entropy().item()==pytest.approx(float(-(left.exp()*left).sum()),abs=1e-6)


def test_null_support_matches_the_inference_presence_rule():
    out=output()
    out.class_logits=torch.tensor([[-2.,2.],[-1.,1.]])
    policy=SpatialPolicy(out,True)
    actions=all_actions()
    assert policy.log_prob(actions)[0].exp().item()==pytest.approx(1-torch.sigmoid(torch.tensor(-2.)).item())
    with pytest.raises(ValueError,match="NULL"):
        SpatialPolicy(out,False).log_prob(actions)
    torch.manual_seed(42)
    sampled=policy.sample(20000)
    assert sampled.present.float().mean().item()==pytest.approx(torch.sigmoid(torch.tensor(-2.)).item(),abs=.01)


def test_frozen_rollouts_produce_coordinate_and_foreground_gradients():
    torch.manual_seed(42)
    out=output(3)
    question=SimpleNamespace(kind="det",categories=["target"],metadata={"allow_empty_target":True},
                             target={"boxes":[[.1,.1,.9,.9]],"labels":[0]})
    objective=SpatialRLCDObjective({"group_size":128})
    rollout=objective.rollout(out,out,question)
    before=rollout.old_log_prob.clone()
    loss,metrics=objective.loss(out,question,rollout)
    assert torch.isfinite(loss)
    loss.backward()
    assert out.coordinate_logits.grad.abs().sum()>0
    assert out.class_logits.grad.abs().sum()>0
    assert not rollout.old_log_prob.requires_grad and not rollout.advantages.requires_grad
    assert torch.equal(rollout.old_log_prob,before)
    assert metrics["sample_iou"]>=0 and metrics["zero_advantage_group"]==0


def test_constant_reward_group_has_zero_policy_advantage():
    out=SpatialOutput(torch.zeros(2,2,requires_grad=True),torch.zeros(2,4,2,requires_grad=True))
    question=SimpleNamespace(kind="det",categories=["target"],metadata={},target={"boxes":[],"labels":[]})
    objective=SpatialRLCDObjective()
    rollout=objective.rollout(out,out,question)
    assert torch.equal(rollout.advantages,torch.zeros_like(rollout.advantages))
    _,metrics=objective.loss(out,question,rollout)
    assert metrics["policy_loss"]==0 and metrics["zero_advantage_group"]==1


def test_wrong_null_cannot_outscore_an_uncertain_wrong_positive_box():
    out=SpatialOutput(torch.zeros(2,2),torch.zeros(2,4,2))
    question=SimpleNamespace(kind="det",categories=["target"],metadata={"allow_empty_target":True},
                             target={"boxes":[[.1,.1,.9,.9]],"labels":[0]})
    torch.manual_seed(42)
    rollout=SpatialRLCDObjective({"group_size":128}).rollout(out,out,question)
    assert rollout.rewards[~rollout.actions.present].max()<rollout.rewards[rollout.actions.present].min()


def tiny_spatial_case(tmp_path):
    from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration
    from valen.modeling.factory import build_model,optimizer_groups
    config=Qwen3_5Config(text_config=dict(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        layer_types=["full_attention"],rope_parameters={"rope_type":"default","rope_theta":10000.,"mrope_section":[2,1,1]}),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=32,
            patch_size=16,spatial_merge_size=2,temporal_patch_size=2,num_position_embeddings=16),
        image_token_id=5,video_token_id=6,vision_start_token_id=7,vision_end_token_id=8)
    base=tmp_path/"base";Qwen3_5ForConditionalGeneration(config).save_pretrained(base)
    settings=dict(architecture="qwen_spatial",task="det",model_path=str(base),dtype="bf16",device="cpu",
                  stage="joint",finetuning_type="full",attn_implementation="sdpa",gradient_checkpointing=True,
                  spatial_width=32,spatial_heads=4,spatial_queries=2,spatial_layers=1,coordinate_bins=16)
    question=SimpleNamespace(inputs={"input_ids":torch.tensor([[7,5,8,9,10,11]]),
        "attention_mask":torch.ones(1,6,dtype=torch.long),"pixel_values":torch.randn(4,1536),
        "image_grid_thw":torch.tensor([[1,2,2]]),"mm_token_type_ids":torch.tensor([[0,1,0,0,0,0]])},
        category_spans=[(3,4)],decision_position=5,grid=(1,1),kind="det",categories=["target"],metadata={},
        target={"boxes":[[.1,.2,.7,.8]],"labels":[0]})
    return settings,question


def test_spatial_rlcd_updates_language_visual_merger_and_head(tmp_path):
    from valen.modeling.factory import build_model,optimizer_groups
    settings,question=tiny_spatial_case(tmp_path)
    model=build_model(settings).train()
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    objective=SpatialRLCDObjective({"group_size":128})
    prediction=model(question)
    rollout=objective.rollout(prediction,prediction,question)
    loss,_=objective.loss(prediction,question,rollout)
    loss.backward();torch.optim.AdamW(optimizer_groups(model,settings)).step()
    for prefix in ["head.","backbone.language_model.","backbone.visual.patch_embed.",
                   "backbone.visual.blocks.","backbone.visual.merger."]:
        assert any(name.startswith(prefix) and p.grad is not None and p.grad.abs().sum()>0 and
                   not torch.equal(p,before[name]) for name,p in model.named_parameters()),prefix


def test_lora_rlcd_loads_merged_sft_then_updates_only_adapters_head_and_merger(tmp_path):
    import json,random
    from safetensors.torch import save_file
    from valen.modeling.factory import build_model,optimizer_groups
    from valen.training.checkpoint import file_sha256,save_checkpoint,load_checkpoint
    from valen.training.spatial_rlcd import initialized_policy,frozen_backbone_matches_reference
    settings,question=tiny_spatial_case(tmp_path)
    source=build_model(dict(settings,stage='warmup',finetuning_type='lora',gradient_checkpointing=False))
    # Make the merged SFT base observably different from the original Qwen base.
    weight=source.backbone.language_model.layers[0].mlp.up_proj.weight
    original=weight.detach().clone()
    with torch.no_grad():weight.add_(.125)
    package=tmp_path/'merged-lora';package.mkdir()
    save_file({k:v.detach().cpu().contiguous() for k,v in source.state_dict().items()},package/'model.safetensors')
    (package/'export_manifest.json').write_text(json.dumps(dict(finetuning_type='lora',
        files_sha256={'model.safetensors':file_sha256(package/'model.safetensors')},
        reload_validation={'independent_process':True})))
    config=dict(settings,finetuning_type='lora',method='rlcd',reference_export=str(package),
                initialization_variant='lora_merged',lora_rank=2,lora_alpha=4,
                head_lr=.001,lora_lr=.001,merger_lr=.001)
    context=SimpleNamespace(primary=True,barrier=lambda:None)
    model,reference=initialized_policy(config,context)
    groups=optimizer_groups(model,config)
    assert {g['name'] for g in groups}=={'head','lora','merger'}
    assert frozen_backbone_matches_reference(model,reference)
    base_weight=model.backbone.language_model.base_model.model.layers[0].mlp.up_proj.base_layer.weight
    assert torch.equal(base_weight,weight) and not torch.equal(base_weight,original)
    with torch.no_grad():
        expected=reference.eval()(question);actual=model.eval()(question)
    torch.testing.assert_close(actual.class_logits,expected.class_logits,rtol=0,atol=0)
    before={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
    model.train();objective=SpatialRLCDObjective({'group_size':128})
    prediction=model(question);rollout=objective.rollout(prediction,reference(question),question)
    loss,_=objective.loss(prediction,question,rollout);loss.backward()
    optimizer=torch.optim.AdamW(groups);optimizer.step()
    for group in ('head','lora','merger'):
        def member(name):
            return name.startswith('head.') if group=='head' else 'lora_' in name if group=='lora' else '.visual.merger.' in name
        assert any(member(name) and p.grad is not None and p.grad.abs().sum()>0 and
                   not torch.equal(p,before[name]) for name,p in model.named_parameters() if name in before),group
    assert frozen_backbone_matches_reference(model,reference)
    checkpoint=tmp_path/'checkpoint'
    save_checkpoint(checkpoint,model,optimizer,config,{'step':1},random.Random(42))
    restored,_=initialized_policy(config,context,make_reference=False)
    load_checkpoint(checkpoint,restored)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
    assert frozen_backbone_matches_reference(restored,reference)
