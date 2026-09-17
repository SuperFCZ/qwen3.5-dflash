from __future__ import annotations

import unittest
from unittest.mock import patch

from dflash_bench.client import (
    RequestError,
    _chunk_text,
    chat_completion,
    parse_sse_lines,
)


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(
            [
                b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
                b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n',
                b'data: {"choices":[],"usage":{"completion_tokens":2}}\n',
                b"data: [DONE]\n",
            ]
        )


class ClientTests(unittest.TestCase):
    def test_parse_sse(self) -> None:
        lines = [
            b": keep-alive\n",
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
            b'data: {"choices":[],"usage":{"completion_tokens":1}}\n',
            b"data: [DONE]\n",
        ]
        chunks = parse_sse_lines(lines)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(_chunk_text(chunks[0]), "hello")

    def test_reasoning_and_content_are_both_preserved(self) -> None:
        chunk = {"choices": [{"delta": {"reasoning_content": "think", "content": "answer"}}]}
        self.assertEqual(_chunk_text(chunk), "thinkanswer")

    def test_bad_json_is_an_error(self) -> None:
        with self.assertRaises(RequestError):
            parse_sse_lines([b"data: nope\n"])

    @patch("urllib.request.urlopen", return_value=FakeResponse())
    def test_chat_completion_collects_stream_and_usage(self, _urlopen) -> None:
        result = chat_completion(
            base_url="http://localhost:8000",
            model="model",
            prompt_id="p1",
            prompt="hello",
            repetition=0,
            max_tokens=2,
            temperature=0.0,
            seed=0,
            timeout_s=1,
        )
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNotNone(result.tpot_s)


if __name__ == "__main__":
    unittest.main()
