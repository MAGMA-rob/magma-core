# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import requests
import json

from .client_base import LLMClientBase

class OllamaClient(LLMClientBase):
    """
    Client for a Ollama Server
    """
    
    def _send(self, data, url, keep_messages=False):
        for _ in range(self.config.max_retry):
            try:
                r = requests.post(
                    url,
                    headers=self.config.headers,
                    data=json.dumps(data),
                    timeout=self.config.timeout
                )
                msg = r.json()["choices"][0]["message"]
                return msg if keep_messages else msg["content"]
            except Exception as e:
                print("Error when posting request to Ollama: " + str(e))
                continue
        raise RuntimeError("Ollama unreachable")

    def send_prompt(self, model, prompt, max_tokens):
        data = {
            "model": model if model is not None else self.config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        return self._send(data, self.config.endpoint)

    def send_messages(self, model, messages, max_tokens, keep_messages=False):
        data = {
            "model": model if model is not None else self.config.default_model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        return self._send(
            data,
            self.config.endpoint,
            keep_messages
        )
