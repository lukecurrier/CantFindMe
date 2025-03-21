import os

from mistralai import Mistral, UserMessage, SystemMessage

from openai import OpenAI
import anthropic
from together import Together
from dotenv import load_dotenv

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()

class Client:
    def __init__(self):
        pass

    def get_completion(self, system: str, message: str, **generate_args):
        pass


class MistralClient(Client):
    def __init__(self, api_key, model=None):
        super().__init__()
        api_key = api_key or os.environ["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=api_key)
        self.model = model or "mistral-medium"

    def get_completion(self, system: str, message: str, **generate_args):
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(UserMessage(content=message))

        if "model" not in generate_args:
            generate_args["model"] = self.model

        chat_response = self.client.chat.complete(
            messages=messages,
            **generate_args
        )
        return chat_response.choices[0].message.content


class OpenAIClient(Client):
    def __init__(self, api_key, model=None, url=None):
        super().__init__()
        api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=api_key, base_url=url)
        self.model = model or "gpt-4"

    @retry(wait=wait_random_exponential(min=6, max=100), stop=stop_after_attempt(5))
    def get_completion(self, system: str, message: str, **generate_args):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content":  message})
        if "model" not in generate_args:
            generate_args["model"] = self.model
        chat_response = self.client.chat.completions.create(
            messages=messages,
            **generate_args
        )
        return chat_response.choices[0].message.content


class AnthropicClient(Client):
    def __init__(self, api_key, model=None):
        super().__init__()
        api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or "claude-3-opus-20240229"

    @retry(wait=wait_random_exponential(min=6, max=100), stop=stop_after_attempt(5))
    def get_completion(self, system: str, message: str, **generate_args):
        messages = []
        messages.append({"role": "user", "content": message})

        model = self.model
        if "model" in generate_args:
            model = generate_args["model"]

        generate_args["model"] = model

        if "temperature" not in generate_args:
            generate_args["temperature"] = 0.3

        response = self.client.messages.create(
            messages=messages,
            max_tokens=4096,
            model = model,
            system = system,
            temperature=generate_args["temperature"]
        )
        return response.content[0].text


class TogetherClient(Client):
    def __init__(self, api_key, model=None):
        super().__init__()
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.client = Together(api_key=api_key)
        self.model = model or "meta-llama/Llama-3-8b-chat-hf"

    @retry(wait=wait_random_exponential(min=6, max=100), stop=stop_after_attempt(5))
    def get_completion(self, system: str, message: str, **generate_args):
        messages = [{"role": "user", "content": message}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        if "model" not in generate_args:
            generate_args["model"] = self.model
        
        if "temperature" not in generate_args:
            generate_args["temperature"] = 1.0

        response = self.client.chat.completions.create(
            model=generate_args["model"],
            messages=messages,
            temperature=generate_args["temperature"]
        )
        return response.choices[0].message.content


def client_from_args(client_str: str, **client_args):
    api_key = client_args.get("api_key")
    model = client_args.get("model")

    if client_str == "mistral":
        return MistralClient(api_key=api_key, model=model)

    elif client_str == "openai":
        return OpenAIClient(api_key=api_key, model=model)

    elif client_str == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)

    elif client_str == "together":
        return TogetherClient(api_key=api_key, model=model)

    else:
        raise ValueError(f"{client_str} is not a supported client - try one of ['mistral', 'openai', 'anthropic', 'together']")