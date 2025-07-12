import os
import json
from openai import OpenAI
from abc import ABC, abstractmethod
from anthropic import Anthropic

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
            model="claude-3-opus-20240229",
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
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=messages,
            system=system_message
        )
        return response.content[0].text

class LlamaModel(LanguageModel):
    """
    Wrapper for a local Llama model.
    """
    def __init__(self, language: str = "english"):
        super().__init__(language)
        # In a real implementation, you would load the local Llama model here.
        print("Llama model initialized (placeholder).")

    def generate_response(self, hate_speech_text: str) -> str:
        template = self.prompt_templates["response_generation"]
        if "user" in template:
            prompt = template["user"].format(hate_speech_text=hate_speech_text)
        else:
            prompt = template["prompt"].format(hate_speech_text=hate_speech_text)
        # This is a placeholder for the actual call to the local Llama model.
        return f"This is a placeholder response from Llama to: '{hate_speech_text}'"

    def classify_response(self, hate_speech_text: str, response_text: str) -> str:
        template = self.prompt_templates["classification"]
        if "user" in template:
            prompt = template["user"].format(hate_speech_text=hate_speech_text, response_text=response_text)
        else:
            prompt = template["prompt"].format(hate_speech_text=hate_speech_text, response_text=response_text)
        # This is a placeholder for the actual call to the local Llama model.
        return "Constructive" # Placeholder classification
