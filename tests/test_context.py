"""Offline tests with a fake Groq client — no API key or network needed.
Run: python -m unittest discover tests
"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import groq
import httpx

import config
from chatbot import Chatbot, ChatError


class FakeClient:
    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, **kw):
        self.calls.append({"model": model, "messages": messages})
        if self.fail_with:
            raise self.fail_with
        text = "SUMMARY: user is Sam, learning Python" if model == config.SUMMARY_MODEL else "ok"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _req():
    return httpx.Request("POST", "https://api.groq.com/x")


class Tests(unittest.TestCase):
    def test_history_is_sent_each_turn(self):
        c = FakeClient(); bot = Chatbot(c)
        _, convo = bot.send(None, "My name is Sam")
        bot.send(convo.id, "I like Python")
        bot.send(convo.id, "What is my name?")
        sent = c.calls[-1]["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("My name is Sam", [m["content"] for m in sent])  # visible 2+ turns later

    def test_summarises_when_long(self):
        c = FakeClient(); bot = Chatbot(c)
        cid = None
        for i in range(12):
            _, convo = bot.send(cid, f"message {i}"); cid = convo.id
        self.assertTrue(convo.summary)
        self.assertLessEqual(len(convo.messages), config.SUMMARIZE_TRIGGER)
        last_system = [x for x in c.calls if x["model"] == config.MODEL][-1]["messages"][0]["content"]
        self.assertIn("Sam", last_system)  # summary injected into the system prompt

    def test_token_budget_truncates(self):
        c = FakeClient(); bot = Chatbot(c)
        old = config.MAX_CONTEXT_TOKENS; config.MAX_CONTEXT_TOKENS = 300
        try:
            cid = None
            for _ in range(4):
                _, convo = bot.send(cid, "x" * 600); cid = convo.id
            sent = c.calls[-1]["messages"]
            self.assertLess(len(sent), 9)
            self.assertEqual(sent[-1]["role"], "user")
        finally:
            config.MAX_CONTEXT_TOKENS = old

    def test_failure_rolls_back_history(self):
        c = FakeClient(fail_with=groq.APITimeoutError(request=_req())); bot = Chatbot(c)
        convo = bot.get(None)
        with self.assertRaises(ChatError) as cm:
            bot.send(convo.id, "hello")
        self.assertEqual(cm.exception.status, 504)
        self.assertEqual(convo.messages, [])

    def test_error_mapping(self):
        resp = httpx.Response(429, request=_req())
        for exc, status in [
            (groq.APIConnectionError(request=_req()), 503),
            (groq.RateLimitError("rl", response=resp, body=None), 429),
        ]:
            bot = Chatbot(FakeClient(fail_with=exc))
            with self.assertRaises(ChatError) as cm:
                bot.send(None, "hi")
            self.assertEqual(cm.exception.status, status)

    def test_empty_and_oversize_input(self):
        bot = Chatbot(FakeClient())
        with self.assertRaises(ChatError): bot.send(None, "   ")
        with self.assertRaises(ChatError): bot.send(None, "x" * (config.MAX_USER_CHARS + 1))


if __name__ == "__main__":
    unittest.main()
