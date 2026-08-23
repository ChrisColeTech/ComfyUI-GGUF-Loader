"""CPU-only smoke test for nodes_lmstudio.py - no GPU, no running LM Studio.

Monkeypatches ``requests`` to return canned OpenAI-style responses and checks
that LMStudioVisionPrompt.generate() round-trips them correctly, and that a
connection failure raises a clean, actionable error instead of a bare
ConnectionError traceback.

Usage:  python tools/smoke_lmstudio.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nodes_lmstudio as lm  # noqa: E402


def _fake_response(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    if status >= 400:
        resp.raise_for_status.side_effect = lm.requests.exceptions.HTTPError(
            f"{status} error")
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_build_chat_payload_text_only():
    payload = lm.build_chat_payload("hello", "some-model", 0.7, 128, 0)
    assert payload["model"] == "some-model"
    assert payload["messages"][-1]["content"][0] == {"type": "text", "text": "hello"}
    assert len(payload["messages"][-1]["content"]) == 1
    print("[ok] build_chat_payload: text-only, one content block")


def test_build_chat_payload_with_image_and_system_prompt():
    image = torch.zeros(1, 8, 8, 3)  # comfy IMAGE: [B, H, W, C], float 0..1
    payload = lm.build_chat_payload(
        "describe this", "vision-model", 0.5, 64, 1,
        image=image, system_prompt="be terse")
    assert payload["messages"][0] == {"role": "system", "content": "be terse"}
    blocks = payload["messages"][1]["content"]
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    print("[ok] build_chat_payload: image block + system prompt present")


def test_extract_reply_happy_path():
    text = lm.extract_reply({"choices": [{"message": {"content": "hi there"}}]})
    assert text == "hi there"
    print("[ok] extract_reply: happy path")


def test_extract_reply_bad_shape_raises_clean_error():
    try:
        lm.extract_reply({"unexpected": "shape"})
    except RuntimeError as e:
        assert "Unexpected LM Studio response" in str(e)
        print("[ok] extract_reply: bad shape raises a clean RuntimeError")
        return
    raise AssertionError("expected RuntimeError")


def test_generate_round_trip():
    node = lm.LMStudioVisionPrompt()
    canned = {"choices": [{"message": {"content": "[VISUAL]: ...\n---\nhi"}}]}
    with patch.object(lm.requests, "post", return_value=_fake_response(canned)) as post:
        (text,) = node.generate(
            prompt="say hi", base_url="http://localhost:1234/v1",
            model="some-model", temperature=0.7, max_tokens=64, seed=0)
        assert text == canned["choices"][0]["message"]["content"]
        assert post.call_args.args[0] == "http://localhost:1234/v1/chat/completions"
    print("[ok] generate: round-trips a mocked /chat/completions response")


def test_generate_connection_error_is_actionable():
    node = lm.LMStudioVisionPrompt()
    with patch.object(lm.requests, "post",
                      side_effect=lm.requests.exceptions.ConnectionError()):
        try:
            node.generate(prompt="x", base_url="http://localhost:1234/v1",
                          model="m", temperature=0.7, max_tokens=64, seed=0)
        except RuntimeError as e:
            assert "Could not reach LM Studio" in str(e)
            print("[ok] generate: connection failure raises an actionable RuntimeError")
            return
    raise AssertionError("expected RuntimeError")


def test_resolve_model_explicit_wins_no_http_call():
    with patch.object(lm.requests, "get") as get:
        model = lm.resolve_model("http://localhost:1234/v1", "qwen2-vl-7b")
        assert model == "qwen2-vl-7b"
        get.assert_not_called()
    print("[ok] resolve_model: explicit model id skips the /models lookup")


def test_resolve_model_blank_auto_detects_first_loaded():
    canned = {"data": [{"id": "loaded-model-a"}, {"id": "loaded-model-b"}]}
    with patch.object(lm.requests, "get", return_value=_fake_response(canned)):
        model = lm.resolve_model("http://localhost:1234/v1", "")
        assert model == "loaded-model-a"
    print("[ok] resolve_model: blank model auto-detects the first loaded one")


def test_resolve_model_blank_unreachable_raises_actionable_error():
    with patch.object(lm.requests, "get", side_effect=lm.requests.exceptions.ConnectionError()):
        try:
            lm.resolve_model("http://localhost:1234/v1", "")
        except RuntimeError as e:
            assert "could not be reached" in str(e)
            print("[ok] resolve_model: unreachable server + blank model raises actionable error")
            return
    raise AssertionError("expected RuntimeError")


def test_resolve_model_blank_no_loaded_model_raises_actionable_error():
    with patch.object(lm.requests, "get", return_value=_fake_response({"data": []})):
        try:
            lm.resolve_model("http://localhost:1234/v1", "")
        except RuntimeError as e:
            assert "no loaded model" in str(e)
            print("[ok] resolve_model: no loaded model raises actionable error")
            return
    raise AssertionError("expected RuntimeError")


if __name__ == "__main__":
    assert lm.requests is not None, "requests must be installed to run this smoke test"
    test_build_chat_payload_text_only()
    test_build_chat_payload_with_image_and_system_prompt()
    test_extract_reply_happy_path()
    test_extract_reply_bad_shape_raises_clean_error()
    test_generate_round_trip()
    test_generate_connection_error_is_actionable()
    test_resolve_model_explicit_wins_no_http_call()
    test_resolve_model_blank_auto_detects_first_loaded()
    test_resolve_model_blank_unreachable_raises_actionable_error()
    test_resolve_model_blank_no_loaded_model_raises_actionable_error()
    print("[ok] all nodes_lmstudio smoke tests passed")
