import os
import json
from openai import OpenAI
from abc import ABC, abstractmethod
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LanguageModel(ABC):
    """
    Abstract base class for language models.
    """
    def __init__(self, language: str = "english"):
        self.language = language
        self._load_prompts()

    def _load_prompts(self):
        script_dir = os.path.dirname(__file__)
        prompts_file_path = os.path.join(script_dir, "prompt_templates.json")
        with open(prompts_file_path, 'r') as f:
            prompts = json.load(f)

        if self.language not in prompts:
            raise ValueError(f"Language '{self.language}' not found in prompt templates.")
        else:
            self.prompt_templates = prompts[self.language]

    @abstractmethod
    def generate_response(self, hate_speech_text: str) -> str:
        """
        Generates a response to a given hate speech text.
        """
        pass

    @abstractmethod
    def classify_response(self, hate_speech_text: str, response_text: str) -> str:
        """
        Classifies the response to a hate speech text.
        """
        pass

class ChatGPTModel(LanguageModel):
    """
    Wrapper for the ChatGPT model.
    """
    def __init__(self, api_key: str, language: str = "english"):
        super().__init__(language)
        if not api_key:
            raise ValueError("API key is required for ChatGPT.")
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, hate_speech_text: str) -> str:
        template = self.prompt_templates["response_generation"]
        messages = [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"].format(hate_speech_text=hate_speech_text)}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content

    def classify_response(self, hate_speech_text: str, response_text: str) -> str:
        template = self.prompt_templates["classification"]
        messages = [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"].format(hate_speech_text=hate_speech_text, response_text=response_text)}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content

    def generate_responses_batch(self, hate_speech_texts: list[str]) -> list[str]:
        """Generate multiple responses using OpenAI's batch API."""
        import json
        import tempfile
        import time

        template = self.prompt_templates["response_generation"]
        operations = []
        for idx, text in enumerate(hate_speech_texts):
            messages = [
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": template["user"].format(hate_speech_text=text)},
            ]
            operations.append({
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o-mini", "messages": messages},
            })

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as f:
            for op in operations:
                json.dump(op, f)
                f.write("\n")
            tmp_path = f.name

        batch_file = self.client.files.create(file=open(tmp_path, "rb"), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            completion_window="24h",
            endpoint="/v1/chat/completions",
        )

        while True:
            batch = self.client.batches.retrieve(batch.id)
            if batch.status in {"completed", "failed", "expired", "canceled"}:
                break
            time.sleep(1)

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch.id} did not complete successfully: {batch.status}")

        content = self.client.files.retrieve_content(batch.output_file_id)
        results = []
        for line in content.decode("utf-8").splitlines():
            data = json.loads(line)
            results.append(data["response"]["choices"][0]["message"]["content"])
        return results

    def classify_responses_batch(self, hate_speech_texts: list[str], responses: list[str]) -> list[str]:
        """Classify multiple responses using OpenAI's batch API."""
        import json
        import tempfile
        import time

        template = self.prompt_templates["classification"]
        operations = []
        for idx, (hate_text, resp) in enumerate(zip(hate_speech_texts, responses)):
            messages = [
                {"role": "system", "content": template["system"]},
                {
                    "role": "user",
                    "content": template["user"].format(
                        hate_speech_text=hate_text,
                        response_text=resp,
                    ),
                },
            ]
            operations.append({
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o-mini", "messages": messages},
            })

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as f:
            for op in operations:
                json.dump(op, f)
                f.write("\n")
            tmp_path = f.name

        batch_file = self.client.files.create(file=open(tmp_path, "rb"), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            completion_window="24h",
            endpoint="/v1/chat/completions",
        )

        while True:
            batch = self.client.batches.retrieve(batch.id)
            if batch.status in {"completed", "failed", "expired", "canceled"}:
                break
            time.sleep(1)

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch.id} did not complete successfully: {batch.status}")

        content = self.client.files.retrieve_content(batch.output_file_id)
        results = []
        for line in content.decode("utf-8").splitlines():
            data = json.loads(line)
            results.append(data["response"]["choices"][0]["message"]["content"])
        return results

class DeepSeekModel(LanguageModel):
    """
    Wrapper for the DeepSeek model.
    """
    def __init__(self, api_key: str, language: str = "english"):
        super().__init__(language)
        if not api_key:
            raise ValueError("API key is required for DeepSeek.")
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")

    def generate_response(self, hate_speech_text: str) -> str:
        template = self.prompt_templates["response_generation"]
        messages = [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"].format(hate_speech_text=hate_speech_text)}
        ]

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return response.choices[0].message.content

    def classify_response(self, hate_speech_text: str, response_text: str) -> str:
        template = self.prompt_templates["classification"]
        messages = [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"].format(hate_speech_text=hate_speech_text, response_text=response_text)}
        ]

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return response.choices[0].message.content

class ClaudeModel(LanguageModel):
    """
    Wrapper for the Claude model.
    """
    def __init__(self, api_key: str, language: str = "english"):
        super().__init__(language)
        if not api_key:
            raise ValueError("API key is required for Claude.")
        self.api_key = api_key
        self.client = Anthropic(api_key=self.api_key)

    def generate_response(self, hate_speech_text: str) -> str:
        template = self.prompt_templates["response_generation"]
        messages = [
            {"role": "user", "content": template["user"].format(hate_speech_text=hate_speech_text)}
        ]
        system_message = template["system"]

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=messages,
            system=system_message
        )
        return response.content[0].text

    def classify_response(self, hate_speech_text: str, response_text: str) -> str:
        template = self.prompt_templates["classification"]
        messages = [
            {"role": "user", "content": template["user"].format(hate_speech_text=hate_speech_text, response_text=response_text)}
        ]
        system_message = template["system"]

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=messages,
            system=system_message
        )
        return response.content[0].text

    def generate_responses_batch(self, hate_speech_texts: list[str]) -> list[str]:
        """Generate multiple responses using Claude's batch API."""
        import time
        requests = []
        template = self.prompt_templates["response_generation"]
        for i, text in enumerate(hate_speech_texts):
            requests.append(
                Request(
                    custom_id=f"my-{i}-request",
                    params=MessageCreateParamsNonStreaming(
                        model="claude-3-haiku-20240307",
                        max_tokens=1024,
                        system=template["system"],
                        messages=[
                            {"role": "user", "content": template["user"].format(hate_speech_text=text)}
                        ]
                    )
                )
            )

        batch = self.client.messages.batches.create(requests=requests)
        seconds = 0
        while True:
            batch = self.client.beta.messages.batches.retrieve(batch.id)
            print(f"Batch {batch.id} status: {batch.processing_status} Time spend: {seconds} seconds")
            seconds += 1
            if batch.processing_status in {"completed", "expired", "canceled", "failed", "ended"}:
                break
            time.sleep(1)
            

        if batch.processing_status != "ended":
            raise RuntimeError(
                f"Batch {batch.id} did not complete successfully: {batch.processing_status}"
            )

        decoder = self.client.messages.batches.results(batch.id)
        results = []
        for item in decoder:
            if item.result.type == "succeeded":
                results.append(item.result.message.content[0].text)
            elif item.result.type == "errored":
                print(f"{item.result =}")
                print(f"Error type: {item.result.error.type}")
                print(f"Error message: {item.result.error.error.message}")
            else:
                results.append("")
        return results

    def classify_responses_batch(self, hate_speech_texts: list[str], responses: list[str]) -> list[str]:
        """Classify multiple responses using Claude's batch API."""
        import time
        requests = []
        template = self.prompt_templates["classification"]
        for i, (hate_text, resp) in enumerate(zip(hate_speech_texts, responses)):
            requests.append(
                Request(
                    custom_id=f"classify-item-{i}",
                    params=MessageCreateParamsNonStreaming(
                        model="claude-3-haiku-20240307",
                        max_tokens=1024,
                        system=template["system"],
                        messages=[
                            {"role": "user", "content": template["user"].format(
                                hate_speech_text=hate_text, response_text=resp)}
                        ]
                    )
                )
            )

        batch = self.client.messages.batches.create(requests=requests)
        seconds = 0
        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            print(f"Batch {batch.id} status: {batch.processing_status} Time spend: {seconds} seconds")
            seconds += 1
            if batch.processing_status in {"completed", "expired", "canceled", "failed", "ended"}:
                break
            time.sleep(1)

        if batch.processing_status != "ended":
            raise RuntimeError(
                f"Batch {batch.id} did not complete successfully: {batch.processing_status}"
            )

        decoder = self.client.messages.batches.results(batch.id)
        results = []
        for item in decoder:
            if item.result.type == "succeeded":
                results.append(item.result.message.content[0].text)
            elif item.result.type == "errored":
                print(f"{item.result =}")
                print(f"Error type: {item.result.error.type}")
                print(f"Error message: {item.result.error.error.message}")
            else:
                results.append("")
        return results

class HFModel(LanguageModel):
    """
    Wrapper for a local Hugging Face model.
    """
    def __init__(self, model_name: str, language: str = "english"):
        super().__init__(language)

        self.model_name = model_name
        cache_dir = ".cache"
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, cache_dir=cache_dir, torch_dtype="auto", device_map="auto"
        )

        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        if self.generator.tokenizer.pad_token_id is None:
            self.generator.tokenizer.pad_token_id = self.generator.tokenizer.eos_token_id

    def generate_response(self, hate_speech_text: str) -> str:
        template = self.prompt_templates["response_generation"]
        messages = [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"].format(hate_speech_text=hate_speech_text)}
        ]

        outputs = self.generator(messages, max_new_tokens=1024, pad_token_id=self.generator.tokenizer.pad_token_id)
        return outputs[0]["generated_text"][-1]['content']

    def classify_response(self, hate_speech_text: str, response_text: str) -> str:
        template = self.prompt_templates["classification"]
        messages = [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"].format(hate_speech_text=hate_speech_text, response_text=response_text)}
        ]

        outputs = self.generator(messages, max_new_tokens=1024, pad_token_id=self.generator.tokenizer.pad_token_id)
        return outputs[0]["generated_text"][-1]['content']

    def generate_responses_batch(self, hate_speech_texts: list[str]) -> list[str]:
        template = self.prompt_templates["response_generation"]
        messages_batch = []
        for text in hate_speech_texts:
            messages = [
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": template["user"].format(hate_speech_text=text)}
            ]
            messages_batch.append(messages)
        print(f"\nGenerating {len(messages_batch)} responses in batch...")
        outputs = self.generator(messages_batch, max_new_tokens=1024, pad_token_id=self.generator.tokenizer.pad_token_id)
        return [output[0]["generated_text"][-1]['content'] for output in outputs]

    def classify_responses_batch(self, hate_speech_texts: list[str], responses: list[str]) -> list[str]:
        template = self.prompt_templates["classification"]
        messages_batch = []
        for hate_text, resp in zip(hate_speech_texts, responses):
            messages = [
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": template["user"].format(hate_speech_text=hate_text, response_text=resp)}
            ]
            messages_batch.append(messages)

        outputs = self.generator(messages_batch, max_new_tokens=1024, pad_token_id=self.generator.tokenizer.pad_token_id)
        return [output[0]["generated_text"][-1]['content'] for output in outputs]