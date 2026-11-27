"""
End-to-end: ONE conversation that hops Claude (adaptive) -> OpenAI ->
Gemini -> Claude (manual thinking) -> OpenAI, with interrupts during a
tool and during streaming along the way. Every provider runs its real
adapter against a scripted fake transport, and every request it sends is
checked against that provider's documented invariants.
"""
import asyncio
from types import SimpleNamespace as NS

from wire_validators import assert_valid_anthropic, assert_valid_gemini, assert_valid_openai

from agent_loop.core.loop import AgentLoop
from agent_loop.core.state import ConversationState, LoopState
from agent_loop.providers.anthropic_provider import AnthropicProvider
from agent_loop.providers.gemini_provider import GeminiProvider
from agent_loop.providers.openai_provider import OpenAIProvider


class _Stream:
    def __init__(self, events, final, final_name, delay):
        self.events, self.final, self.delay = events, final, delay
        setattr(self, final_name, self._final)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self.events:
            await asyncio.sleep(self.delay)
            yield e

    async def _final(self):
        return self.final


class FakeAnthropic:
    def __init__(self):
        self.script, self.requests = [], []
        self.messages = self

    def stream(self, **kw):
        self.requests.append(kw)
        assert_valid_anthropic(kw["messages"])
        events, final, delay = self.script.pop(0)
        return _Stream(events, final, "get_final_message", delay)


class FakeOpenAI:
    def __init__(self):
        self.script, self.requests = [], []
        self.responses = self

    def stream(self, **kw):
        self.requests.append(kw)
        assert_valid_openai(kw["input"])
        events, final, delay = self.script.pop(0)
        return _Stream(events, final, "get_final_response", delay)


class FakeGemini:
    def __init__(self):
        self.script, self.requests = [], []
        self.aio = NS(models=self)

    async def generate_content_stream(self, **kw):
        self.requests.append(kw)
        assert_valid_gemini(kw["contents"])
        chunks, delay = self.script.pop(0)

        async def gen():
            for c in chunks:
                await asyncio.sleep(delay)
                yield c
        return gen()


# --- scripted responses ---------------------------------------------------------

def claude_tool(tid, cmd, sig):
    final = NS(content=[NS(type="thinking", thinking="I should run a command", signature=sig),
                        NS(type="tool_use", id=tid, name="bash", input={"command": cmd})],
               stop_reason="tool_use", usage=None)
    events = [NS(type="content_block_start", index=0, content_block=NS(type="thinking")),
              NS(type="content_block_delta", index=0, delta=NS(type="thinking_delta", thinking="I should run a command")),
              NS(type="content_block_stop", index=0),
              NS(type="content_block_start", index=1, content_block=NS(type="tool_use", id=tid, name="bash")),
              NS(type="content_block_stop", index=1)]
    return events, final, 0


def claude_text(text):
    final = NS(content=[NS(type="thinking", thinking="ok", signature="S-final"), NS(type="text", text=text)],
               stop_reason="end_turn", usage=None)
    return [NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text=text))], final, 0


def openai_call(call_id, rs):
    final = NS(status="completed", usage=None, output=[
        NS(type="reasoning", id=rs, encrypted_content="ENC-" + rs, summary=[NS(text="reasoning summary")]),
        NS(type="function_call", call_id=call_id, name="file_edit",
           arguments='{"path": "notes.md", "content": "hello"}')])
    return [], final, 0


def openai_text(text, rs):
    final = NS(status="completed", usage=None, output=[
        NS(type="reasoning", id=rs, encrypted_content="ENC-" + rs, summary=[]),
        NS(type="message", content=[NS(type="output_text", text=text)])])
    return [NS(type="response.output_text.delta", delta=text)], final, 0


def gemini_slow_text(words):
    def chunk(text, sig=None):
        return NS(candidates=[NS(content=NS(parts=[NS(text=text, thought=False, thought_signature=sig, function_call=None)]),
                                 finish_reason=None)], usage_metadata=None)
    return [chunk(w + " ") for w in words], 0.05


async def run(loop, text, interrupt_after=None):
    task = asyncio.ensure_future(_consume(loop, text))
    if interrupt_after is not None:
        await asyncio.sleep(interrupt_after)
        assert await loop.interrupt()
    await asyncio.wait([task])
    assert loop.state.state in (LoopState.IDLE, LoopState.DONE)


async def _consume(loop, text):
    async for _ in loop.run_turn(text):
        pass


async def test_one_conversation_across_three_providers_with_interrupts(tools):
    ant, oai, gem = FakeAnthropic(), FakeOpenAI(), FakeGemini()
    providers = {
        "anthropic": AnthropicProvider(ant),
        "openai": OpenAIProvider(oai),
        "gemini": GeminiProvider(gem),
    }
    loop = AgentLoop(ConversationState(active_provider="anthropic", active_model="claude-opus-5-5"), providers, tools)

    # 1. Claude: one tool round-trip, then a slow tool that gets interrupted.
    ant.script += [claude_tool("toolu_1", "echo hi", "S1"), claude_tool("toolu_2", "sleep 5", "S2")]
    await run(loop, "look around", interrupt_after=0.5)
    assert loop.last_interrupt.killed_tools

    # 2. Switch to OpenAI mid-conversation: tool call + answer.
    loop.switch_provider("openai", "gpt-5.5")
    oai.script += [openai_call("call_A", "rs_A"), openai_text("wrote notes", "rs_B")]
    await run(loop, "write notes instead")
    req = oai.requests[-1]
    assert [i["id"] for i in req["input"] if i["type"] == "reasoning"] == ["rs_A"]  # own reasoning only

    # 3. Gemini: interrupted mid-stream.
    loop.switch_provider("gemini", "gemini-3.5-flash")
    gem.script += [gemini_slow_text(["word"] * 30)]
    await run(loop, "summarize everything", interrupt_after=0.3)
    assert loop.state.messages[-2].partial
    first_call = next(p for c in gem.requests[-1]["contents"] for p in c["parts"] if "function_call" in p)
    assert first_call["thought_signature"]  # foreign calls carry the skip sentinel

    # 4. Claude with MANUAL thinking (4.5 family): history full of foreign turns.
    loop.switch_provider("anthropic", "claude-haiku-4-5")
    ant.script += [claude_text("back on claude")]
    await run(loop, "what happened so far?")
    req = ant.requests[-1]
    assert req["thinking"]["type"] == "enabled"
    signed = [b for m in req["messages"] for b in m["content"] if b["type"] == "thinking"]
    assert {b["signature"] for b in signed} == {"S1", "S2"}  # only Claude's own, unmodified

    # 5. And back to OpenAI once more.
    loop.switch_provider("openai", "gpt-5.3-codex")
    oai.script += [openai_text("all good", "rs_C")]
    await run(loop, "thanks")
    assert loop.state.messages[-1].content[-1].text == "all good"
    assert (tools["file_edit"].workdir / "notes.md").read_text() == "hello"
