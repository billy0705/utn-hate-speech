from abc import ABC, abstractmethod

class LanguageModel(ABC):
    """
    Abstract base class for language models.
    """
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

import os
from openai import OpenAI

class ChatGPTModel(LanguageModel):
    """
    Wrapper for the ChatGPT model.
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required for ChatGPT.")
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, hate_speech_text: str) -> str:
        prompt = f"Generate a constructive and respectful response to the following text: '{hate_speech_text}'"
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def classify_response(self, hate_speech_text: str, response_text: str) -> str:
        prompt = f"""
        Given the following hate speech text and the response to it, classify the response into one of the following categories:
        - Constructive
        - Neutral
        - Unhelpful
        - Escalating

        Hate Speech: '{hate_speech_text}'
        Response: '{response_text}'

        Classification:
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class LlamaModel(LanguageModel):
    """
    Wrapper for a local Llama model.
    """
    def __init__(self):
        # In a real implementation, you would load the local Llama model here.
        print("Llama model initialized (placeholder).")

    def generate_response(self, hate_speech_text: str) -> str:
        prompt = f"Generate a constructive and respectful response to the following text: '{hate_speech_text}'"
        # This is a placeholder for the actual call to the local Llama model.
        return f"This is a placeholder response from Llama to: '{hate_speech_text}'"

    def classify_response(self, hate_speech_text: str, response_text: str) -> str:
        prompt = f"""
        Given the following hate speech text and the response to it, classify the response into one of the following categories:
        - Constructive
        - Neutral
        - Unhelpful
        - Escalating

        Hate Speech: '{hate_speech_text}'
        Response: '{response_text}'

        Classification:
        """
        # This is a placeholder for the actual call to the local Llama model.
        return "Constructive" # Placeholder classification
