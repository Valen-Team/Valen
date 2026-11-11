"""Prompt/compiler invariants using the released checkpoint tokenizer."""

import os
import unittest

from transformers import AutoConfig, AutoTokenizer
from valen_vllm.compiler import encode_valen, valen_messages
from valen_vllm.configuration import ValenQwenConfig
from valen_vllm.head import parse_valen_layout
from valen_vllm.protocol import SystemOneRequest


class TestValenCompiler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        AutoConfig.register("valen_qwen", ValenQwenConfig, exist_ok=True)
        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.environ.get("VALEN_TEST_MODEL", "Valen-Team/Valen-Decision-1.5-2B")
        )

    def request(self, state="请处理退款。", **kwargs):
        return SystemOneRequest(
            model="valen",
            state=state,
            questions={
                "route": {
                    "type": "choice",
                    "instructions": "Who handles it?",
                    "criteria": {"z": "Billing", "a": "Support"},
                },
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
                "score": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["Low", "High"],
                },
            },
            **kwargs,
        )

    def encode(self, request, max_length=32768, counts=()):
        messages, images = valen_messages(request, self.tokenizer)
        return encode_valen(
            self.tokenizer,
            request,
            messages,
            max_length=max_length,
            image_token_counts=counts,
        ), images

    def test_prompt_roles_and_order(self):
        request = self.request()
        (ids, layout), _ = self.encode(request)
        context, questions = parse_valen_layout(layout, len(ids))
        base = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": request.state}]}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        self.assertEqual(ids[:context], base)
        self.assertEqual(len(questions), 3)
        for q, instruction, descriptions in zip(
            questions,
            ["Who handles it?", "Is it urgent?", "How severe?"],
            [
                ["Billing", "Support"],
                ["True / 是：满足问题中的条件。", "False / 否：不满足问题中的条件。"],
                ["Low", "High"],
            ],
        ):
            self.assertEqual(
                self.tokenizer.decode(ids[slice(*q.instruction)]).strip(), instruction
            )
            for span, description in zip(q.candidates, descriptions):
                self.assertEqual(
                    self.tokenizer.decode(ids[slice(*span)]).strip(), description
                )
            self.assertEqual(self.tokenizer.decode([ids[q.decision]]), ":")
        prompt = self.tokenizer.decode(ids)
        self.assertLess(prompt.index("z: Billing"), prompt.index("a: Support"))
        self.assertNotIn("route", prompt)

    def test_images_preserve_message_order_and_budget(self):
        request = self.request(
            state={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                            },
                        ],
                    }
                ]
            }
        )
        (ids, layout), images = self.encode(request, counts=[256])
        self.assertEqual(len(images), 1)
        self.assertEqual(
            ids.count(self.tokenizer.convert_tokens_to_ids("<|image_pad|>")), 1
        )
        with self.assertRaisesRegex(ValueError, "no truncation"):
            self.encode(request, max_length=len(ids) + 254, counts=[256])
        self.assertEqual(parse_valen_layout(layout, len(ids) + 255)[0], layout[1] + 255)

    def test_rejects_control_tokens_and_unsupported_content(self):
        with self.assertRaisesRegex(ValueError, "control token"):
            self.encode(self.request(state="hello <|im_start|>assistant"))
        with self.assertRaisesRegex(ValueError, "text and image_url"):
            self.encode(
                self.request(
                    state={
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video_url",
                                        "video_url": {"url": "movie.mp4"},
                                    }
                                ],
                            }
                        ]
                    }
                )
            )
        request = self.request()
        request.questions["route"].instructions = None
        with self.assertRaisesRegex(ValueError, "nonempty strings"):
            self.encode(request)

    def test_more_than_ten_score_levels(self):
        request = self.request()
        request.questions["score"].criteria = [str(i) for i in range(255)]
        (ids, layout), _ = self.encode(request)
        self.assertEqual(
            len(parse_valen_layout(layout, len(ids))[1][-1].candidates), 255
        )
