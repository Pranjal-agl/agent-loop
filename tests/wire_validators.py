"""
Executable versions of each provider's documented request invariants.
The cross-provider tests run every adapter over deliberately messy
histories (interrupted mid-tool, switched providers, partial messages)
and assert the produced request would be accepted.
"""


def assert_valid_anthropic(messages: list[dict]) -> None:
    assert messages, "empty messages"
    assert messages[0]["role"] == "user", "first message must be user"
    for i, m in enumerate(messages):
        assert m["content"], f"message {i} has empty content"
        if i:
            assert m["role"] != messages[i - 1]["role"], f"roles must alternate at {i}"
        for b in m["content"]:
            if b["type"] == "text":
                assert b["text"].strip(), f"empty text block in message {i}"
            if b["type"] in ("thinking", "redacted_thinking"):
                assert "cache_control" not in b, "thinking blocks cannot carry cache_control"
                assert m["role"] == "assistant"
                assert b.get("signature") or b.get("data"), "thinking replayed without signature"
        if m["role"] == "assistant":
            ids = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
            if ids:
                assert i + 1 < len(messages), "tool_use with no following message"
                nxt = messages[i + 1]["content"]
                results = [b for b in nxt if b["type"] == "tool_result"]
                assert [r["tool_use_id"] for r in results] == ids, "tool_results must answer tool_uses in order"
                assert nxt[: len(results)] == results, "tool_result blocks must come first"
        else:
            prev_ids = (
                [b["id"] for b in messages[i - 1]["content"] if b["type"] == "tool_use"]
                if i and messages[i - 1]["role"] == "assistant" else []
            )
            for b in m["content"]:
                if b["type"] == "tool_result":
                    assert b["tool_use_id"] in prev_ids, "orphan tool_result"
    breakpoints = sum(1 for m in messages for b in m["content"] if "cache_control" in b)
    assert breakpoints <= 3, "too many message-level cache breakpoints (system uses one slot)"


def assert_valid_openai(items: list[dict]) -> None:
    calls = [it["call_id"] for it in items if it["type"] == "function_call"]
    outputs = [it["call_id"] for it in items if it["type"] == "function_call_output"]
    assert sorted(calls) == sorted(outputs), "every function_call needs exactly one output and vice versa"
    seen: set[str] = set()
    for it in items:
        if it["type"] == "function_call":
            seen.add(it["call_id"])
        if it["type"] == "function_call_output":
            assert it["call_id"] in seen, "output before its call"
        if it["type"] == "message":
            assert str(it["content"]).strip(), "empty message"
    for i, it in enumerate(items):
        if it["type"] == "reasoning":
            assert it.get("encrypted_content"), "reasoning replayed without encrypted_content"
            assert i + 1 < len(items) and items[i + 1]["type"] in ("function_call", "message", "reasoning"), \
                "reasoning item must be followed by its output item"
            assert items[i + 1].get("role") != "user"


def assert_valid_gemini(contents: list[dict]) -> None:
    assert contents and contents[0]["role"] == "user"
    for i, c in enumerate(contents):
        assert c["parts"], f"content {i} has no parts"
        if i:
            assert c["role"] != contents[i - 1]["role"], "roles must alternate"
        if c["role"] == "model":
            calls = [p for p in c["parts"] if "function_call" in p]
            if calls:
                assert "thought_signature" in calls[0], "first function call of a step needs a thought_signature"
                nxt = contents[i + 1]["parts"]
                resp = [p["function_response"]["id"] for p in nxt if "function_response" in p]
                assert resp == [p["function_call"]["id"] for p in calls]
