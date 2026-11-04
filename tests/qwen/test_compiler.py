from copy import deepcopy
from pathlib import Path
import random
import pytest
import torch
from valen.data.compiler import Compiler
from valen.data.schema import read_jsonl


def test_chinese_jsonl_in_ascii_locale():
    import locale
    previous = locale.setlocale(locale.LC_CTYPE)
    try:
        locale.setlocale(locale.LC_CTYPE, "C")
        assert len(read_jsonl("data/smoke/train.jsonl")) == 6
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)


def test_pre_audited_media_skips_hash_io_but_keeps_same_inputs(compiler, monkeypatch):
    from valen.data.schema import read_jsonl
    record = next(r for r in read_jsonl("data/smoke/train.jsonl") if r.get("assets") and r["assets"][0]["path"].endswith(".png"))
    messages, media = compiler._messages(record["request"]["state"])
    cached = Compiler(compiler.processor, compiler.media_root, verified_media_hashes={m["path"]: m["sha256"] for m in media})
    def unexpected_read(path):
        raise AssertionError("Already-audited media should not be rehashed during timing")
    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    assert cached._messages(record["request"]["state"]) == (messages, media)


@pytest.fixture(scope="module")
def compiler():
    from transformers import AutoProcessor
    path = Path("models/Qwen3.5-2B")
    if not (path / "tokenizer.json").exists():
        pytest.skip("Prepare the official processor first")
    return Compiler(AutoProcessor.from_pretrained(path, local_files_only=True), "data/smoke", 8192)


def test_real_tokenizer_endpoints_and_score_isolation(compiler):
    record = read_jsonl("data/smoke/text.jsonl")[0]
    compiled = compiler.compile(record, random.Random(7))
    assert len(compiled.questions) == 3
    for q in compiled.questions:
        for branch in q.branches:
            ids = branch.inputs["input_ids"][0]
            assert branch.decision_position == len(ids) - 1
            assert compiler.tokenizer.decode(ids[-3:]).endswith("Decision:")
            assert all(p < branch.decision_position for p in branch.candidate_positions)
        if q.kind == "score":
            assert len(q.branches) == 3
            for i, branch in enumerate(q.branches):
                text = compiler.tokenizer.decode(branch.inputs["input_ids"][0])
                for j, description in enumerate(q.descriptions):
                    assert (description in text) == (i == j)
                assert len(branch.candidate_positions) == 1
        if q.kind == "choice":
            assert q.target[q.keys.index("red")] == 1


def test_role_spans_preserve_original_tokenization(compiler):
    record = read_jsonl("data/smoke/text.jsonl")[0]
    state = compiler.compile(record)
    for q in state.questions:
        instruction = record["request"]["questions"][q.qid]["instructions"]
        for i, branch in enumerate(q.branches):
            ids = branch.inputs["input_ids"][0]
            a, b = branch.instruction_span
            assert compiler.tokenizer.decode(ids[a:b]).strip() == instruction.strip()
            pairs = list(zip(q.keys, q.descriptions))
            group = [pairs[i]] if q.kind == "score" else pairs
            pieces = ["<|im_start|>user\n", f"Task: {q.kind}\nQuestion: {instruction}\nCandidates:\n"]
            for (key, description), (a, b) in zip(group, branch.candidate_spans):
                assert compiler.tokenizer.decode(ids[a:b]).strip() == description.strip()
                pieces.extend([(key + ": " if q.kind != "score" else "") + description, "\n"])
            pieces.append("Decision:")
            old_suffix = [token for text in pieces for token in compiler.tokenizer.encode(text, add_special_tokens=False)]
            assert ids[branch.context_span[1]:].tolist() == old_suffix


def test_question_id_does_not_enter_prompt(compiler):
    record = read_jsonl("data/smoke/text.jsonl")[0]
    first = compiler.compile(record).questions[0].branches[0].inputs["input_ids"]
    changed = deepcopy(record)
    changed["request"]["questions"] = {"NEVER_SHOW_THIS_ID": record["request"]["questions"]["category"]}
    changed["targets"] = {}
    second = compiler.compile(changed).questions[0].branches[0].inputs["input_ids"]
    assert torch.equal(first, second)


def test_no_silent_truncation(compiler):
    small = Compiler(compiler.processor, max_length=5)
    with pytest.raises(ValueError, match="no truncation"):
        small.compile(read_jsonl("data/smoke/text.jsonl")[0])


def test_missing_labels_skip_before_media(compiler):
    record = read_jsonl("data/smoke/text.jsonl")[-1]
    record["request"]["state"] = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "missing.png"}}]}]}
    assert compiler.compile(record, labeled_only=True).questions == []


def test_multimodal_expansion(compiler):
    for record in read_jsonl("data/smoke/train.jsonl"):
        if not isinstance(record["request"]["state"], dict):
            continue
        compiled = compiler.compile(record)
        assert not compiled.questions[0].is_text
        branch = compiled.questions[0].branches[0]
        assert branch.inputs["input_ids"].shape == branch.inputs["attention_mask"].shape
        assert any(k.startswith("pixel_values") for k in branch.inputs)
        assert compiled.media


def test_video_rope_matches_expanded_sequence(compiler):
    # Regression: Transformers 5.3.0 exhausted the video-grid iterator on frame 2.
    from transformers import AutoConfig, Qwen3_5Model
    config = AutoConfig.from_pretrained("models/Qwen3.5-2B", local_files_only=True)
    with torch.device("meta"):
        backbone = Qwen3_5Model(config)
    record = next(r for r in read_jsonl("data/smoke/train.jsonl") if r["group_id"] == "synthetic-video")
    branch = compiler.compile(record).questions[0].branches[0]
    inputs = branch.inputs
    position_ids, _ = backbone.get_rope_index(input_ids=inputs["input_ids"],
        mm_token_type_ids=inputs["mm_token_type_ids"], video_grid_thw=inputs["video_grid_thw"],
        attention_mask=inputs["attention_mask"])
    assert position_ids.shape == (3, 1, branch.decision_position + 1)


def test_noul_rubric_is_folded_into_the_instructions():
    """A custom Noul rubric must reach the model, matching JevalKit's render_questions."""
    from valen.data.schema import fold_noul_criteria
    question = {"type": "noul", "instructions": "Does this record qualify?",
                "criteria": {"true": "The threshold is met", "false": "The threshold is not met"}}
    folded = fold_noul_criteria(question)
    assert "criteria" not in folded
    assert folded["instructions"] == ("Does this record qualify?\nTrue criterion: The threshold is met"
                                      "\nFalse criterion: The threshold is not met")
    # Unchanged without a rubric, and for other kinds.
    plain = {"type": "noul", "instructions": "Is it red?"}
    assert fold_noul_criteria(plain) is plain
    choice = {"type": "choice", "instructions": "Pick", "criteria": {"a": "A"}}
    assert fold_noul_criteria(choice) is choice


def test_noul_rubric_reaches_the_compiled_prompt(compiler):
    question = {"type": "noul", "instructions": "Does this record qualify?",
                "criteria": {"true": "The threshold is met", "false": "The threshold is not met"}}
    state = compiler.compile({"group_id": "g", "request": {"state": "s", "questions": {"q": question}}})
    text = compiler.tokenizer.decode(state.questions[0].branches[0].inputs["input_ids"][0])
    assert "True criterion: The threshold is met" in text
