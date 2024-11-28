"""
# Author: https://github.com/pyautoml
# Topic: Agents (Models) with multiple custom tools.

# Description: Create and use custom tools with paid and open-source models.
1. In this tutorial, we do NOT use any paid services like SerpAPI. 
However, to utilize commercial models, you will need to set up accounts and provide API keys.
2. It is advised, but not necessary, to run code examples on a GPU for faster inference.
3. Remember to create locally .env file and add OpenAI apikey:
    openai_apikey=...
4. Make sure you have downloaded Ollama locally. Then, in the Command Prompt (CMD), run: ollama pull llama3.2.
- After that, run: ollama list.
- You should see the output in the terminal.
  C:\Users\...>ollama list
    NAME                       ID              SIZE      MODIFIED
    llama3.2:latest            a80c4f17acd5    2.0 GB    1 day ago

# Agenda
1. Creating an Ollama-based Agent
2. Creating an OpenAI-based Agent
3. Creating Custom Tools from Scratch
4. Adapting Custom Tools to Different Types of Models

Have fun! 🐱
"""

import os
import json
import openai
import ollama
from dotenv import load_dotenv
from typing import Any, List, Optional
from memory import ChatMessage, BaseMemory
from tools import run_callable, today_is_tool, weather_tool, day_of_week_tool


load_dotenv(interpolate=False)


class BaseChatModel:
    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0.0,
        keep_alive: int = -1,
        format: Optional[str] = None,
        max_stored_memory_messages: int = 50,
    ) -> None:

        self.client = client
        self.model = model
        self.format = format
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.memory = BaseMemory(max_size=max_stored_memory_messages)
        self.system = """You are a helpful AI assistant. When using tools, always provide a natural, complete response using the information gathered.
        Format your response as a short, coherent sentence."""

    def message(self, human: str, ai: str) -> ChatMessage:
        return ChatMessage(message={"human": human, "ai": ai})


class OllamaChatModel(BaseChatModel):
    """http://localhost:11434 is the default Ollama port serving API."""

    def __init__(self, tools: list[dict], model: str = "llama3.2") -> None:
        self.model = model
        self.tools = tools
        self.client = ollama.Client(host="http://localhost:11434")
        super().__init__(client=self.client, model=self.model)

    def extract(self, tool_call) -> list:
        """Extract and execute tool call"""
        data = []

        if not isinstance(tool_call, list):
            tool_call = [tool_call]

        for tool in tool_call:
            func_name = tool.function.name
            if isinstance(tool.function.arguments, str):
                func_arguments = json.loads(tool.function.arguments)
            else:
                func_arguments = tool.function.arguments
            result = run_callable(func_name, func_arguments)
            data.append(result)
        return data

    def response(self, user_prompt: str, system_message: str = None) -> ollama.ChatResponse:
        messages = [
            {
                "role": "system",
                "content": system_message if system_message else self.system,
            }
        ]

        for msg in self.memory.get():
            if isinstance(msg, ChatMessage):
                messages.extend(
                    [
                        {"role": "user", "content": msg.human},
                        {"role": "assistant", "content": msg.ai},
                    ]
                )

        messages.append({"role": "user", "content": user_prompt})

        return self.client.chat(
            model=self.model,
            messages=messages,
            format=self.format,
            keep_alive=self.keep_alive,
            tools=self.tools,
        )

    def chat(self, system_message: str = None, save_chat: bool = False) -> None:
        system_message = system_message if system_message else self.system

        while True:
            print("current memory: ", self.memory.get(), "\n\n")

            user_prompt = input("User: ")
            if user_prompt == "bye":
                self.memory.add(self.message(human=user_prompt, ai="Bye"))
                if save_chat:
                    self.memory.save(model_name=str(self.model))
                self.memory.clear()
                print("AI: Bye")
                break

            response = self.response(user_prompt, system_message)

            # If there are tool calls, process them and get final response
            if hasattr(response.message, "tool_calls") and response.message.tool_calls:
                collected_data = {}

                for tool_call in response.message.tool_calls:
                    result = self.extract(tool_call)
                    collected_data[tool_call.function.name] = result

                final_prompt = (
                    f"Based on the following information:\n"
                    f"{collected_data}"
                    f"Please provide a natural response to the original question: '{user_prompt}'"
                )

                final_response = self.client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Sumup your response before sending back. Make is short and concise"
                            + " "
                            + system_message,
                        },
                        {"role": "user", "content": final_prompt},
                    ],
                )
                response_content = final_response.message.content
            else:
                # If no tool calls, use the original response
                response_content = response.message.content

            if response_content:
                print(f"AI: {response_content}", end="\n\n")
                self.memory.add(self.message(human=user_prompt, ai=response_content))


class OpenAIChatModel(BaseChatModel):
    def __init__(
        self,
        api_key: str,
        tools: List[dict],
        model: str = "gpt-4",
        timeout: Optional[int] = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        self.tools = tools
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        # store model configuration is needed
        # self.model_config = {"model": model, "temperature": temperature, "tools": tools}
        super().__init__(client=self.client, model=self.model)

    def extract(self, tool_calls) -> list:
        """Extract and execute tool calls"""
        data = []

        if not isinstance(tool_calls, list):
            tool_call = [tool_calls]

        for tool_call in tool_calls:
            func_name = tool_call.function.name
            func_arguments = json.loads(tool_call.function.arguments)
            result = run_callable(func_name, func_arguments)
            data.append(result)
        return data

    def response(self, user_prompt: str, system_message: str = None) -> openai.types.Completion:
        messages = [
            {
                "role": "system",
                "content": system_message if system_message else self.system,
            }
        ]
        for msg in self.memory.get():
            if isinstance(msg, ChatMessage):
                messages.extend(
                    [
                        {"role": "user", "content": msg.human},
                        {"role": "assistant", "content": msg.ai},
                    ]
                )
        messages.append({"role": "user", "content": user_prompt})

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            temperature=self.temperature,
        )

    def chat(self, system_message: str = None, save_chat: bool = False) -> None:
        system_message = system_message if system_message else self.system

        while True:
            print("current memory: ", self.memory.get(), "\n\n")
            user_prompt = input("User: ")
            if user_prompt.lower() == "bye":
                self.memory.add(self.message(human=user_prompt, ai="Bye"))
                if save_chat:
                    self.memory.save(model_name=str(self.model))
                self.memory.clear()
                print("AI: Bye")
                break

            response = self.response(user_prompt, system_message)

            if response.choices[0].message.tool_calls:
                collected_data = {}

                for tool_call in response.choices[0].message.tool_calls:
                    results = self.extract([tool_call])
                    collected_data[tool_call.function.name] = results

                final_prompt = (
                    f"Based on the following information:\n"
                    f"{collected_data}\n"
                    f"Please provide a natural response to the original question: '{user_prompt}'"
                )

                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Summarize your response before sending back. Make it short and concise. "
                            + system_message,
                        },
                        {"role": "user", "content": final_prompt},
                    ],
                )
                response_content = final_response.choices[0].message.content
            else:
                response_content = response.choices[0].message.content

            if response_content:
                print(f"AI: {response_content}", end="\n\n")
                self.memory.add(self.message(human=user_prompt, ai=response_content))


def load_model(model: str = None) -> OllamaChatModel | OpenAIChatModel:
    tools = [today_is_tool, weather_tool, day_of_week_tool]
    match model:
        case "ollama":
            return OllamaChatModel(tools=tools)
        case "openai":
            return OpenAIChatModel(api_key=os.environ.get("openai_apikey"), tools=tools)
        case _:
            return OllamaChatModel(tools=tools)


def main():
    model = load_model("openai")
    model.chat(save_chat=True)


if __name__ == "__main__":
    main()
