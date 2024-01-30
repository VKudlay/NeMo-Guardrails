# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_core.tools import Tool

from types import SimpleNamespace
from nemoguardrails.streaming import StreamingHandler

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.logging.callbacks import LoggingCallbackHandler
from nemoguardrails.context import explain_info_var, llm_call_info_var
# from .llmrails_lite import LLMRails
# from nemoguardrails import RailsConfig

import asyncio
import sys
import os
import threading
import queue

# import multiprocessing
import asyncio
import queue


class GenRunThread(threading.Thread):
# class GenRunThread(multiprocessing.Process):
    # https://stackoverflow.com/a/75094151/5003309
    def __init__(self, coro=None, agen=None):
        assert coro or agen, "coro or agen must be specified"
        self.coro = coro
        self.agen = agen
        if agen:
            self.q = queue.Queue()
        self.result = None
        self.exception = None
        super().__init__()

    def get_result(self):
        self.join()
        return self.result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join()
        if self.exception:
            raise self.exception
        del self

    def run(self):
        fn = self.run_async_coro if self.coro else self.run_async_gen
        try: asyncio.run(fn())
        except Exception as e: self.exception = e

    async def run_async_coro(self):
        try: self.result = await asyncio.gather(self.coro)
        except Exception as e: self.exception = e

    async def run_async_gen(self):
        try:
            async for item in self.agen: 
                self.q.put(item)
        except Exception as e: 
            self.q.put(e)
        finally:
            self.q.put(None)  # Signal the end of the stream or exception

    # def start(self) -> None:
    #     print(f"Starting Thread {self.name}")
    #     return super().start()

    # def join(self) -> None:
    #     print(f"Joining Thread {self.name}")
    #     return super().join()
    
    # def __del__(self) -> None:
    #     print(f"Deleting Thread {self.name}")


def run_in_threaded_loop(coro):
    try: loop = asyncio.get_running_loop()
    except RuntimeError: loop = None
    if loop and loop.is_running():
        with GenRunThread(coro=coro) as thread:
            return thread.get_result()
    else:
        return asyncio.gather(coro)

def run_stream_in_threaded_loop(agen):
    with GenRunThread(agen=agen) as thread:
        while True:
            item = thread.q.get()
            if item is None: break
            if isinstance(item, Exception): 
                raise item
            yield item

# Code to clear all previous threads
def clear_all_threads():
    for thread in threading.enumerate():
        if isinstance(thread, GenRunThread) and thread.is_alive():
            thread.join()


class RunnableRails(Runnable[Input, Output]):
    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[BaseLanguageModel] = None,
        tools: Optional[List[Tool]] = None,
        passthrough: bool = True,
        passthrough_runnable: Optional[Runnable] = None,
        input_key: str = "input",
        output_key: str = "output",
    ) -> None:
        self.llm = llm
        self.passthrough = passthrough
        self.passthrough_runnable = passthrough_runnable
        self.passthrough_user_input_key = input_key
        self.passthrough_bot_output_key = output_key
        self.handler = StreamingHandler()

        config.streaming = True
        # We override the config passthrough.
        config.passthrough = passthrough

        self.rails = LLMRails(config=config, llm=llm)
        # self.rails.explain_info = ExplainInfo()
        # explain_info_var.set(self.rails.explain_info)
        # self.llm.callbacks = [LoggingCallbackHandler()]

        if tools:
            # When tools are used, we disable the passthrough mode.
            self.passthrough = False

            for tool in tools:
                self.rails.register_action(tool, tool.name)

        # If we have a passthrough Runnable, we need to register a passthrough fn
        # that will call it
        if self.passthrough_runnable:
            self._init_passthrough_fn()

    def _init_passthrough_fn(self):
        """Initialize the passthrough function for the LLM rails instance."""

        async def passthrough_fn(context: dict, events: List[dict]):
            # First, we fetch the input from the context
            _input = context.get("passthrough_input")
            print("IN:", repr(_input))

            output_msg = None
            dict_output = {}

            coro_gen = self.passthrough_runnable.astream(_input)
            async for chunk in coro_gen:
                if isinstance(chunk, dict):
                    chunk_body = chunk.get(self.passthrough_bot_output_key, chunk)
                    dict_output = {**chunk, **dict_output}
                else:
                    chunk_body = getattr(chunk, "content", chunk)
                    if chunk:
                        await self.handler.push_chunk(chunk)
                output_msg = (chunk) if (not output_msg) else (output_msg + chunk)

            print("OUT:", repr(dict_output), repr(output_msg))

            output_body = getattr(output_msg, "content", output_msg)
            return output_body, (dict_output if dict_output else output_msg)
            

        self.rails.llm_generation_actions.passthrough_fn = passthrough_fn

    def __or__(self, other):
        self.passthrough_runnable = other
        self.passthrough = True
        self._init_passthrough_fn()
        return self

    @property
    def InputType(self) -> Any:
        return Any

    @property
    def OutputType(self) -> Any:
        """The type of the output of this runnable as a type annotation."""
        return Any

    def _transform_input_to_rails_format(self, _input):
        messages = []

        if self.passthrough and self.passthrough_runnable and isinstance(_input, (str, dict)):
            # First, we add the raw input in the context variable $passthrough_input
            if isinstance(_input, dict):
                text_input = _input.get(self.passthrough_user_input_key, _input)
            else:
                text_input = getattr(_input, "content", _input)

            messages = [
                {
                    "role": "context",
                    "content": {
                        "passthrough_input": _input,
                        # We also set all the input variables as top level context variables
                        **(_input if isinstance(_input, dict) else {}),
                    },
                },
                {
                    "role": "user",
                    "content": text_input,
                },
            ]

        else:
            if isinstance(_input, ChatPromptValue):
                for msg in _input.messages:
                    if isinstance(msg, AIMessage):
                        messages.append({"role": "assistant", "content": msg.content})
                    elif isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
            elif isinstance(_input, StringPromptValue):
                messages.append({"role": "user", "content": _input.text})
            elif isinstance(_input, dict):
                # If we're provided a dict, then the `input` key will be the one passed
                # to the guardrails.
                if "input" not in _input:
                    raise Exception("No `input` key found in the input dictionary.")

                # TODO: add support for putting the extra keys as context
                user_input = _input["input"]
                if isinstance(user_input, str):
                    messages.append({"role": "user", "content": user_input})
                elif isinstance(user_input, list):
                    # If it's a list of messages
                    for msg in user_input:
                        assert "role" in msg
                        assert "content" in msg
                        messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )
                else:
                    raise Exception(
                        f"Can't handle input of type {type(user_input).__name__}"
                    )

                if "context" in _input:
                    if not isinstance(_input["context"], dict):
                        raise ValueError(
                            "The input `context` key for `RunnableRails` must be a dict."
                        )
                    messages = [{"role": "context", "content": _input["context"]}] + messages

            else:
                raise Exception(f"Can't handle input of type {type(_input).__name__}")

        return messages

    async def _generate_results(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
    ):
        rails_coroutine = self.rails.generate_async(
            prompt, messages, self.handler, return_context=True
        )
        result_context = await asyncio.gather(rails_coroutine)
        await self.handler.queue.put(result_context[0])

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,  ## ?? What is this for?
        return_full: Optional[bool] = False,
        **kwargs: Optional[Any],
    ) -> Output:
        input_messages = self._transform_input_to_rails_format(input)
        self.handler = StreamingHandler()
        run_in_threaded_loop(self._generate_results(messages=input_messages))
        while self.handler:
            chunk = await self.handler.queue.get()
            if isinstance(chunk, tuple):
                result, context = chunk
                if "passthrough_output" not in context:
                    print("BOT Failed")
                    break
                else:
                    print("BOT Passed")
                    break
            elif chunk: 
                yield chunk

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,  ## ?? What is this for?
        return_full: Optional[bool] = False,
        **kwargs: Optional[Any],
    ) -> Output:
        """Invoke this runnable synchronously."""
        return run_in_threaded_loop(self.ainvoke(input, config, return_full))[0]

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,  ## ?? What is this for?
        return_full: Optional[bool] = False,
        **kwargs: Optional[Any],
    ) -> Output:
        """Invoke this runnable synchronously."""
        output = None
        async for token in self.astream(input, config, return_full):
            output = (token) if (not output) else (output + token)
        return output

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        return_full: Optional[bool] = False,
        **kwargs: Optional[Any],
    ) -> Output:
        for token in run_stream_in_threaded_loop(self.astream(input, config, return_full)):
            yield token
