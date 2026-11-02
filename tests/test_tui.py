"""Headless tests for the Textual frontend (no real agent/model)."""
import asyncio

from textual.widgets import Input, Markdown

from jarvis.tui import ConfirmScreen, EventMessage, OptimusTUI, UserMessage


class FakeAgent:
    def __init__(self):
        self.history = [{"role": "system", "content": "s"}]
        self.confirm = None
        self.on_event = None
        self.calls = []

    def ask(self, text, max_steps=None, on_delta=None):
        self.calls.append(text)
        if self.on_event:
            self.on_event({"type": "route", "provider": "cloud",
                           "model": "fake", "reason": "test"})
            self.on_event({"type": "tool", "name": "shell_run",
                           "args": {"cmd": "ls"}})
        if on_delta:
            on_delta("Hello ")
            on_delta("**world**")
        return "Hello **world**"


class ConfirmingAgent(FakeAgent):
    def ask(self, text, max_steps=None, on_delta=None):
        self.approved = self.confirm("run a command", "shell_run",
                                     {"cmd": "ls"}, False)
        return f"approved={self.approved}"


def test_streams_answer_without_tool_activity():
    async def run():
        agent = FakeAgent()
        app = OptimusTUI(agent=agent)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            prompt = app.query_one("#prompt", Input)
            prompt.value = "hi there"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause(0.2)

            assert agent.calls == ["hi there"]
            assert len(app.query(UserMessage)) == 1
            assert len(app.query(EventMessage)) == 0
            markdown = app.query_one(Markdown)
            assert markdown._markdown == "Hello **world**"
            assert app._busy is False
            app.exit()

    asyncio.run(run())


def test_confirmation_dialog_approves_tool_call():
    async def run():
        agent = ConfirmingAgent()
        app = OptimusTUI(agent=agent)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            prompt = app.query_one("#prompt", Input)
            prompt.value = "do it"
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause(0.3)
            await app.workers.wait_for_complete()
            await pilot.pause(0.2)
            assert agent.approved is True
            app.exit()

    asyncio.run(run())
