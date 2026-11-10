"""Valen-Decision-1.5 in the vLLM and SGLang plugins, checked against the native code."""
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from valen.data.compilers.qwen import Compiler
from valen.modeling.qwen.block_attention import forward_kwargs, segment_mask
from valen.modeling.qwen.grounding import BoxFeatures, GroundingHead, decode

ROOT = Path(__file__).resolve().parents[2]
START, END, VISION_START, IMAGE, VISION_END = range(200000, 200005)
SPECIAL = {"<|im_start|>": START, "<|im_end|>": END, "<|vision_start|>": VISION_START,
           "<|image_pad|>": IMAGE, "<|vision_end|>": VISION_END}
GRID = (2, 3)


class Tokenizer:
    """Characters as tokens, control strings as single ids. / 字符即 token，控制符为单个 id。"""
    all_special_tokens = list(SPECIAL)
    added_tokens_decoder = {}
    pattern = re.compile("(" + "|".join(map(re.escape, SPECIAL)) + ")")

    def __call__(self, text, **kwargs):
        ids, offsets, cursor = [], [], 0
        for piece in self.pattern.split(text):
            tokens = [SPECIAL[piece]] if piece in SPECIAL else [ord(c) for c in piece]
            width = len(piece) if piece in SPECIAL else 1
            for token in tokens:
                ids.append(token)
                offsets.append((cursor, cursor + width))
                cursor += width
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]

    def convert_tokens_to_ids(self, token):
        return SPECIAL[token]

    def apply_chat_template(self, messages, expand=False, **kwargs):
        ids = []
        for message in messages:
            ids += self.encode("<|im_start|>" + message["role"] + "\n")
            for item in message["content"]:
                if item["type"] == "text":
                    ids += self.encode(item["text"])
                else:
                    ids += [VISION_START] + [IMAGE] * (GRID[0] * GRID[1] if expand else 1) + [VISION_END]
            ids += self.encode("<|im_end|>\n")
        return ids


class Processor:
    tokenizer = Tokenizer()
    image_token_id = IMAGE

    def apply_chat_template(self, messages, **kwargs):
        ids = torch.tensor([self.tokenizer.apply_chat_template(messages, expand=True)])
        images = sum(item["type"] == "image" for m in messages for item in m["content"])
        base = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if images:
            base["image_grid_thw"] = torch.tensor([[1, GRID[0] * 2, GRID[1] * 2]] * images)
        return base


QUESTIONS = {
    "route": {"type": "choice", "instructions": "Who handles it?", "criteria": {"z": "Billing", "a": "Support"}},
    "urgent": {"type": "noul", "instructions": "Is it urgent?",
               "criteria": {"true": "A deadline is named.", "false": "No deadline is named."}},
    "where": {"type": "bbox", "instructions": "the invoice number"},
    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["Low", "Middle", "High"]},
    "stamp": {"type": "bbox", "instructions": "the stamp", "image": 1},
}


def native(tmp_path, questions, images=2):
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    content = [{"type": "image_url", "image_url": {"url": str(path)}} for _ in range(images)]
    content.append({"type": "text", "text": "An invoice was charged twice."})
    state = {"messages": [{"role": "user", "content": content}]}
    compiled = Compiler(Processor(), execution="shared_state").compile(
        {"request": {"state": state, "questions": questions}})
    return state, compiled


def reference_segments(mask, branch):
    return forward_kwargs(mask, [branch], "cpu")["valen_segments"][0]


def expand(ids):
    """The engine's placeholder expansion. / 引擎对图像占位符的展开。"""
    return [t for token in ids for t in ([IMAGE] * (GRID[0] * GRID[1]) if token == IMAGE else [token])]


@pytest.fixture
def sglang():
    """The plugin's pure modules, with the two engine imports they name."""
    pooler, utils = ModuleType("sglang.srt.layers.pooler"), ModuleType("sglang.srt.utils")
    pooler.EmbeddingPoolerOutput = lambda embeddings: SimpleNamespace(embeddings=embeddings)
    utils.ImageData = lambda url, detail="auto": SimpleNamespace(url=url, detail=detail)
    stubs = {name: ModuleType(name) for name in ("sglang", "sglang.srt", "sglang.srt.layers")}
    stubs.update({pooler.__name__: pooler, utils.__name__: utils})
    with patch.dict(sys.modules, stubs), patch.object(sys, "path", [str(ROOT / "serving/sglang/src"), *sys.path]):
        from valen_sglang import blocks, compiler, grounding, head
        yield SimpleNamespace(blocks=blocks, compiler=compiler, grounding=grounding, head=head)


@pytest.fixture
def vllm():
    with patch.object(sys, "path", [str(ROOT / "serving/vllm/src"), *sys.path]):
        from valen_vllm import answers, blocks, compiler, configuration, grounding, head, protocol
        yield SimpleNamespace(answers=answers, blocks=blocks, compiler=compiler, configuration=configuration,
                              grounding=grounding, head=head, protocol=protocol)


def request(questions, state):
    def question(q):
        criteria = q.get("criteria")
        if q["type"] == "noul" and criteria:
            criteria = SimpleNamespace(**criteria)
        return SimpleNamespace(type=q["type"], instructions=q["instructions"], criteria=criteria,
                               image=q.get("image", 0))
    return SimpleNamespace(state=state, images=[], questions={k: question(q) for k, q in questions.items()})


def dense(q, k, v, mask, scale):
    groups = q.shape[1] // k.shape[1]
    k, v = (x.transpose(0, 1).repeat_interleave(groups, 0) for x in (k, v))
    return F.scaled_dot_product_attention(q.transpose(0, 1), k, v, attn_mask=mask, scale=scale).transpose(0, 1)


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_block_attention_matches_the_reference_mask(engine, tmp_path, request):
    plugin = request.getfixturevalue(engine)
    _, compiled = native(tmp_path, QUESTIONS)
    branch = compiled.questions[0].branches[0]
    torch.manual_seed(0)
    length = branch.sequence_length
    q, k, v = torch.randn(length, 4, 8), torch.randn(length, 2, 8), torch.randn(length, 2, 8)
    for mask in ("bbox_blocks", "question_blocks"):
        segments = reference_segments(mask, branch)
        visible = segment_mask(segments[None], torch.tensor([branch.context_span[1]]), torch.tensor([length]))[0]
        torch.testing.assert_close(plugin.blocks.block_attention(q, k, v, segments, 0.3),
                                   dense(q, k, v, visible, 0.3), atol=1e-5, rtol=1e-5)
    causal = torch.ones(length, length).tril().bool()
    torch.testing.assert_close(plugin.blocks.block_attention(q, k, v, None, 0.3),
                               dense(q, k, v, causal, 0.3), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mask", ["bbox_blocks", "question_blocks", "block_bidirectional"])
def test_sglang_prompt_and_layout_match_the_native_compiler(sglang, tmp_path, mask):
    state, compiled = native(tmp_path, QUESTIONS)
    body = request(QUESTIONS, state)
    messages, images = sglang.compiler.valen_messages(body, Tokenizer())
    count = GRID[0] * GRID[1]
    ids, layout = sglang.compiler.encode_valen(
        Tokenizer(), body, messages, max_length=4096, image_token_counts=[count] * len(images),
        fold=True, mask=mask, image_grids=[GRID] * len(images))
    branches = [q.branches[0] for q in compiled.questions]
    assert expand(ids) == branches[0].inputs["input_ids"][0].tolist()
    length = branches[0].sequence_length
    context, questions = sglang.head.parse_valen_layout(layout, length)
    assert context == branches[0].context_span[1]
    for question, branch in zip(questions, branches):
        assert question.instruction == branch.instruction_span
        assert list(question.candidates) == branch.candidate_spans
        assert question.decision == branch.decision_position
        if branch.image_positions is None:
            assert question.image is None
        else:
            start, rows, cols = question.image
            assert list(range(start, start + rows * cols)) == branch.image_positions.tolist()
            assert (rows, cols) == branch.image_grid
    segments = sglang.head.valen_segments(layout, length)
    if mask == "block_bidirectional":
        # Transformers block_sequence_ids semantics: the whole suffix is one shared block.
        assert segments.tolist() == [-1] * context + [0] * (length - context)
    else:
        assert torch.equal(segments, reference_segments(mask, branches[0]))


def test_sglang_causal_layout_is_unchanged_and_rejects_rubrics(sglang, tmp_path):
    questions = {k: q for k, q in QUESTIONS.items() if q["type"] != "bbox"}
    questions["urgent"] = {"type": "noul", "instructions": "Is it urgent?"}
    state, _ = native(tmp_path, questions, images=0)
    body = request(questions, state)
    messages, _ = sglang.compiler.valen_messages(body, Tokenizer())
    ids, layout = sglang.compiler.encode_valen(Tokenizer(), body, messages, max_length=4096, image_token_counts=[])
    _, parsed = sglang.head.parse_valen_layout(layout, len(ids))
    assert layout == sglang.head.pack_valen_layout(len(ids), layout[1], parsed)
    assert sglang.head.valen_segments(layout, len(ids)) is None
    with pytest.raises(ValueError):
        sglang.head.parse_valen_layout(layout + [1, 3], len(ids))
    body = request(QUESTIONS, state)
    with pytest.raises(ValueError, match="omit criteria"):
        sglang.compiler.encode_valen(Tokenizer(), body, messages, max_length=4096, image_token_counts=[])


@pytest.mark.parametrize("images", [0, 2])
def test_vllm_prompt_and_token_segments_match_the_native_compiler(vllm, tmp_path, images):
    questions = {k: q for k, q in QUESTIONS.items() if images or q["type"] != "bbox"}
    state, compiled = native(tmp_path, questions, images)
    body = request(questions, state)
    body.questions["urgent"].criteria = questions["urgent"]["criteria"]
    messages, _ = vllm.compiler.valen_messages(body, Tokenizer())
    counts = [GRID[0] * GRID[1]] * images
    short, layout = vllm.compiler.encode_valen(Tokenizer(), body, messages, max_length=4096, image_token_counts=counts,
                                               fold=True, mask="bbox_blocks", image_grids=[GRID] * images)
    branch = compiled.questions[0].branches[0]
    ids = torch.tensor(expand(short))
    assert torch.equal(ids, branch.inputs["input_ids"][0])
    encode = Tokenizer().encode
    glue = [encode(text) for text in vllm.blocks.GLUE]
    expected = reference_segments("bbox_blocks", branch)
    assert torch.equal(vllm.blocks.token_segments(ids, glue, encode=encode), expected)
    # The endpoint checks the token reading against the compiler's blocks before expansion.
    shift = len(ids) - len(short)
    assert torch.equal(vllm.blocks.token_segments(torch.tensor(short), glue, encode=encode),
                       vllm.head.valen_segments(layout, len(ids))[shift:])
    assert torch.equal(vllm.head.valen_segments(layout, len(ids)), expected)
    _, parsed = vllm.head.parse_valen_layout(layout, len(ids))
    for q, native_question in zip(parsed, compiled.questions):
        branch = native_question.branches[0]
        assert q.decision == branch.decision_position
        if native_question.kind == "bbox":
            start, rows, cols = q.image
            assert (rows, cols) == tuple(branch.image_grid)
            assert branch.image_positions.tolist() == list(range(start, start + rows * cols))
    shared = vllm.blocks.token_segments(ids, glue, bidirectional=True)
    context = branch.context_span[1]
    assert shared.tolist() == [-1] * context + [0] * (len(ids) - context)
    assert vllm.blocks.token_segments(torch.zeros(9, dtype=torch.long), glue) is None
    # A header copied into question text moves the token reading away from the compiler's blocks.
    if images:
        body.questions["urgent"].instructions = "Question 3\nTask: bbox"
        body.questions["where"].type, body.questions["where"].criteria = "noul", None
        short, layout = vllm.compiler.encode_valen(Tokenizer(), body, messages, max_length=4096,
                                                   image_token_counts=counts, fold=True, mask="bbox_blocks",
                                                   image_grids=[GRID] * images)
        assert not torch.equal(vllm.blocks.token_segments(torch.tensor(short), glue, encode=encode),
                               vllm.head.valen_segments(layout, len(short) + shift)[shift:])
    with pytest.raises(ValueError, match="omit criteria"):
        vllm.compiler.encode_valen(Tokenizer(), body, messages, max_length=4096, image_token_counts=[])


def test_vllm_protocol_and_config(vllm):
    Question = vllm.protocol.Question
    rubric = {"true": "Yes", "false": "No"}
    assert Question(type="noul", instructions="x", criteria=rubric).criteria == rubric
    for criteria in (["wrong"], {"true": "Yes"}):
        with pytest.raises(ValueError):
            Question(type="noul", instructions="x", criteria=criteria)
    Config = vllm.configuration.ValenQwenConfig
    legacy, current = Config(), Config(valen_full_attention_mask="bbox_blocks", valen_grounding={"width": 8, "hidden": 8})
    assert (legacy.valen_full_attention_mask, legacy.valen_grounding, legacy.valen_decision_version) == ("causal", None, "1.0")
    assert current.valen_decision_version == "1.5"


def test_vllm_answers_read_five_bbox_values_between_candidate_logits(vllm):
    body = request({k: QUESTIONS[k] for k in ("where", "severity")}, {})
    values = [0.1, 0.2, 0.5, 0.6, 0.7, 0.0, 0.0, 0.0]
    result = vllm.answers.answers(body, values)
    assert result["where"] == vllm.grounding.box_answer(values[:5])
    assert list(result["severity"]["probabilities"].values()) == pytest.approx([1 / 3] * 3)
    with pytest.raises(RuntimeError):
        vllm.answers.answers(body, values[:-1])
    question = vllm.protocol.Question(type="bbox", instructions="the stamp", image=1)
    assert (question.image, question.criteria) == (1, None)
    with pytest.raises(ValueError):
        vllm.protocol.Question(type="bbox", instructions="x", criteria=["a", "b"])


def test_sglang_grounding_head_and_pooler_match_the_native_head(sglang):
    torch.manual_seed(0)
    reference = GroundingHead(12, 8, 16)
    head = sglang.grounding.GroundingHead(12, 8, 16)
    head.load_state_dict(reference.state_dict())
    hidden = torch.randn(40, 12)
    rows, cols = 3, 4
    expected = decode(reference(BoxFeatures(hidden[35], hidden[2:14], (rows, cols))))
    values = head(hidden[35], hidden[2:14], (rows, cols))
    assert sglang.grounding.box_answer(values.tolist()) == dict(expected, type="bbox")

    Question = sglang.head.ValenQuestion
    questions = [Question((20, 22), ((22, 24), (24, 26)), 34), Question((26, 28), (), 35, (2, rows, cols)),
                 Question((28, 30), ((30, 32),), 36)]
    layout = sglang.head.pack_valen_layout(40, 16, questions, [0] * 4 + [1] * 6 + [2] * 2 + [1] * 4 + [0] * 2
                                           + [1, 2, 1] + [0] * 3)
    assert sglang.head.parse_valen_layout(layout, 40) == (16, questions)
    mixer = sglang.head.ValenMixerHead(12, {"head_width": 8, "head_layers": 1, "head_token_hidden_dim": 4,
                                            "head_channel_hidden_dim": 16})
    batch = SimpleNamespace(batch_size=1, decision_layouts=[layout], extend_seq_lens_cpu=[40],
                            extend_prefix_lens_cpu=None)
    output = sglang.head.ValenPooler(mixer, head)(hidden, batch).embeddings[0]
    assert output.shape == (2 + 5 + 1,)
    torch.testing.assert_close(output[2:7], values)
    plain = [questions[0], questions[2]]
    batch.decision_layouts = [sglang.head.pack_valen_layout(40, 16, plain)]
    logits = sglang.head.ValenPooler(mixer)(hidden, batch).embeddings[0]
    torch.testing.assert_close(torch.cat((output[:2], output[7:])), logits)


def test_sglang_image_grids_follow_the_processor_resize(sglang):
    processor = SimpleNamespace(patch_size=16, merge_size=2, size={"shortest_edge": 65536, "longest_edge": 1048576})
    assert sglang.grounding.image_grids(processor, [(480, 640), (64, 64)]) == [(15, 20), (8, 8)]
