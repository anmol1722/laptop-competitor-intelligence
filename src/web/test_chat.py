"""Hosted chat contract and failure tests; no API key, GPU or downloads required.

Run: python -m unittest discover -s src/web -p 'test_chat.py' -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient
from src.web import api
from web.chat_controls import ChatControls
from hosted_llm import ENDPOINT, GroqLLM, HostedLLMError
import config
import rag
import requests

TEST_KEY = "test-only-credential-do-not-expose"
MODEL = "openai/gpt-oss-120b"


def provider_response(status=200, payload=None, retry_after=None):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload if payload is not None else {
        "choices": [{"message": {"content": "The Test Laptop costs $999 [P1]."},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 15},
    }).encode()
    response._content_consumed = True
    if retry_after is not None:
        response.headers["Retry-After"] = str(retry_after)
    return response


class GroqTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"GROQ_API_KEY": TEST_KEY})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.llm = GroqLLM(MODEL)

    def test_completion_uses_existing_messages_and_returns_generation_stats(self):
        messages = [{"role": "system", "content": "Only use the evidence."},
                    {"role": "user", "content": "Price? [P1] $999"}]
        with patch("hosted_llm.requests.post", return_value=provider_response()) as post:
            text, stats = self.llm.chat(messages, max_new_tokens=420)
        args, kwargs = post.call_args
        self.assertEqual(args[0], ENDPOINT)
        self.assertEqual(kwargs["json"]["messages"], messages)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer " + TEST_KEY)
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["timeout"], (10, 60))
        self.assertFalse(kwargs["json"]["include_reasoning"])
        self.assertEqual(kwargs["json"]["max_completion_tokens"], 1444)
        self.assertIn("[P1]", text)
        self.assertEqual(stats["prompt_tokens"], 50)
        self.assertEqual(stats["new_tokens"], 15)
        self.assertFalse(stats["truncated"])

    def test_errors_are_sanitized_and_use_retry_metadata(self):
        cases = [(401, "provider_auth_error", 503, None),
                 (403, "provider_auth_error", 503, None),
                 (429, "provider_rate_limited", 429, 13),
                 (500, "provider_unavailable", 503, 15),
                 (400, "provider_request_rejected", 502, None),
                 (302, "provider_request_rejected", 502, None)]
        for status, code, public_status, retry in cases:
            with self.subTest(status=status), patch("hosted_llm.requests.post", return_value=
                    provider_response(status, {"error": {"message": TEST_KEY}}, 12.5)):
                with self.assertRaises(HostedLLMError) as raised:
                    self.llm.chat([])
                exc = raised.exception
                self.assertEqual(exc.code, code)
                self.assertEqual(exc.status_code, public_status)
                self.assertEqual(exc.retry_after, retry)
                self.assertNotIn(TEST_KEY, str(exc))

    def test_network_errors_do_not_expose_request_details(self):
        for failure, code in [(requests.Timeout(TEST_KEY), "provider_timeout"),
                              (requests.ConnectionError(TEST_KEY), "provider_unreachable")]:
            with self.subTest(code=code), patch("hosted_llm.requests.post", side_effect=failure):
                with self.assertRaises(HostedLLMError) as raised:
                    self.llm.chat([])
                self.assertEqual(raised.exception.code, code)
                self.assertNotIn(TEST_KEY, str(raised.exception))

    def test_invalid_answers_are_not_successes(self):
        for payload in ({}, {"choices": []}, {"choices": [{"message": {"content": ""}}]},
                        {"choices": [{"message": {"content": None}}]}):
            with self.subTest(payload=payload), patch("hosted_llm.requests.post",
                    return_value=provider_response(payload=payload)):
                with self.assertRaises(HostedLLMError) as raised:
                    self.llm.chat([])
                self.assertEqual(raised.exception.code, "provider_invalid_response")

    def test_invalid_retry_after_uses_a_finite_default(self):
        with patch("hosted_llm.requests.post", return_value=provider_response(429, retry_after="nan")):
            with self.assertRaises(HostedLLMError) as raised:
                self.llm.chat([])
        self.assertEqual(raised.exception.retry_after, 60)

    def test_truncation_is_preserved(self):
        response = provider_response(payload={
            "choices": [{"message": {"content": "Partial answer"}, "finish_reason": "length"}],
        })
        with patch("hosted_llm.requests.post", return_value=response):
            _, stats = self.llm.chat([])
        self.assertTrue(stats["truncated"])


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.controls = ChatControls(limits=((2, 60), (3, 3600)), cache_ttl=10,
                                     cache_size=2, clock=lambda: self.now)

    def test_both_rate_windows_expire_and_clients_are_independent(self):
        self.assertEqual(self.controls.check_rate("alice"), 0)
        self.assertEqual(self.controls.check_rate("alice"), 0)
        self.assertEqual(self.controls.check_rate("alice"), 60)
        self.assertEqual(self.controls.check_rate("bob"), 0)
        self.now = 60
        self.assertEqual(self.controls.check_rate("alice"), 0)
        self.now = 120
        self.assertEqual(self.controls.check_rate("alice"), 3480)
        self.now = 3600
        self.assertEqual(self.controls.check_rate("alice"), 0)

    def test_cache_is_bounded_copied_and_expires(self):
        self.controls.cache_answer((1,), {"answer": ["original"]})
        cached = self.controls.get_cached((1,))
        cached["answer"].append("mutation")
        self.assertEqual(self.controls.get_cached((1,)), {"answer": ["original"]})
        self.controls.cache_answer((2,), {})
        self.controls.cache_answer((3,), {})
        self.assertIsNone(self.controls.get_cached((1,)))
        self.now = 10
        self.assertIsNone(self.controls.get_cached((2,)))

    def test_provider_cooldown_never_shortens(self):
        self.controls.defer_provider(60)
        self.controls.defer_provider(10)
        self.assertEqual(self.controls.retry_after(), 60)
        self.now = 61
        self.assertEqual(self.controls.retry_after(), 0)


class ChatAPITests(unittest.TestCase):
    def setUp(self):
        for context in (patch.dict(os.environ, {"GROQ_API_KEY": TEST_KEY}),
                        patch.object(config, "LLM_PROVIDER", "groq"),
                        patch.object(config, "GROQ_MODEL", MODEL)):
            context.start()
            self.addCleanup(context.stop)
        self.manager = api.LLMManager()
        spec = rag.parse_question("What is the price?", [])
        evidence = [rag.Evidence("[P1]", "product", "[P1] Test Laptop. Price: $999.",
                                 {"parent_asin": "TEST123", "title": "Test Laptop", "price": 999})]
        self.agent = rag.RagAgent(retriever=SimpleNamespace(brands=[]), llm=GroqLLM(MODEL))
        self.agent.retrieve = Mock(return_value=(spec, evidence, [], {"n_evidence": 1}))
        for context in (patch.object(api, "LLM", self.manager),
                        patch.object(api, "CHAT_CONTROLS", ChatControls()),
                        patch.object(rag, "get_agent", return_value=self.agent),
                        patch.object(api, "_gpu_memory_used_mib", side_effect=AssertionError("GPU used"))):
            context.start()
            self.addCleanup(context.stop)
        self.client = TestClient(api.app)
        self.addCleanup(self.client.close)

    def ask(self, question="What is the price?", **kwargs):
        return self.client.post("/api/chat", json={"question": question, **kwargs})

    def test_hosted_chat_keeps_grounding_and_caches_repeated_questions(self):
        with patch("hosted_llm.requests.post", return_value=provider_response()) as post:
            first = self.ask()
            second = self.ask("What  is the price?")
        self.assertEqual(first.status_code, 200, first.text)
        data = first.json()
        self.assertTrue(data["audit"]["grounded"])
        self.assertEqual(data["audit"]["citation_rate"], 1.0)
        self.assertEqual(data["n_evidence"], 1)
        self.assertTrue(data["audit"]["citations"])
        self.assertFalse(data["cached"])
        self.assertTrue(second.json()["cached"])
        self.assertEqual(second.json()["question"], "What  is the price?")
        post.assert_called_once()
        self.assertIn("[P1]", post.call_args.kwargs["json"]["messages"][-1]["content"])
        self.assertEqual(self.manager.n_answers, 1)
        self.assertNotIn(TEST_KEY, first.text)

    def test_missing_key_disables_chat_but_keeps_ui_available(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("hosted_llm.requests.post") as post:
            status = self.client.get("/api/chat/status").json()
            response = self.ask()
        self.assertEqual(status["state"], "unavailable")
        self.assertFalse(status["accepting_input"])
        self.assertIsNone(status["gpu"])
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "chat_not_configured")
        self.assertEqual(self.client.get("/").status_code, 200)
        post.assert_not_called()
        self.assertFalse(self.manager.gen_lock.locked())

    def test_provider_limit_sets_retry_header_and_prevents_repeat_api_calls(self):
        with patch("hosted_llm.requests.post", return_value=provider_response(429, retry_after=30)) as post:
            first = self.ask()
            second = self.ask("Another question?")
        for response in (first, second):
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers["Retry-After"], "30")
            self.assertEqual(response.json()["error"]["code"], "provider_rate_limited")
        post.assert_called_once()
        self.assertFalse(self.manager.gen_lock.locked())

    def test_auth_errors_are_sanitized_and_status_becomes_unavailable(self):
        with patch("hosted_llm.requests.post", return_value=provider_response(401, {"error": TEST_KEY})):
            response = self.ask()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(TEST_KEY, response.text)
        self.assertEqual(self.client.get("/api/chat/status").json()["state"], "unavailable")

    def test_timeouts_allow_a_later_retry(self):
        with patch("hosted_llm.requests.post", side_effect=requests.Timeout(TEST_KEY)):
            response = self.ask()
        self.assertEqual(response.status_code, 504)
        self.assertTrue(response.json()["error"]["retryable"])
        self.assertFalse(self.manager.gen_lock.locked())
        with patch("hosted_llm.requests.post", return_value=provider_response()):
            self.assertEqual(self.ask().status_code, 200)

    def test_client_limit_cannot_be_bypassed_with_a_forwarding_header(self):
        with patch("hosted_llm.requests.post", return_value=provider_response()) as post:
            for _ in range(3):
                self.assertEqual(self.ask().status_code, 200)
            response = self.client.post("/api/chat", json={"question": "What is the price?"},
                                        headers={"X-Forwarded-For": "different-client"})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"]["code"], "client_rate_limited")
        post.assert_called_once()

    def test_busy_generation_returns_a_retry_without_using_provider(self):
        self.manager.gen_lock.acquire()
        try:
            with patch("hosted_llm.requests.post") as post:
                response = self.ask()
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.json()["error"]["code"], "generation_in_progress")
            post.assert_not_called()
        finally:
            self.manager.gen_lock.release()

    def test_local_generation_still_checks_gpu_before_loading(self):
        with patch.object(config, "LLM_PROVIDER", "local"):
            manager = api.LLMManager()
        self.agent.llm = Mock(model_name=config.LLM_MODEL)
        with patch.object(manager, "_wait_for_gpu", return_value=(True, {})) as gpu:
            self.assertTrue(manager.ensure_ready()[0])
        gpu.assert_called_once()
        self.agent.llm.load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
