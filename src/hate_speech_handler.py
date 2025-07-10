import os
from models import LanguageModel, ChatGPTModel, LlamaModel

def load_env_vars(env_path=".env"):
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key] = value.strip('\'"')
    return env_vars

class HateSpeechHandler:
    """
    A class to handle hate speech classification and response generation using different LLMs.
    """

    def __init__(self, model: LanguageModel):
        """
        Initializes the HateSpeechHandler with a specified model.

        Args:
            model (LanguageModel): An instance of a class that inherits from LanguageModel.
        """
        self.model = model

    def generate_response(self, hate_speech_text: str) -> str:
        """
        Generates a response to a given hate speech text using the selected model.

        Args:
            hate_speech_text (str): The hate speech text.

        Returns:
            str: The generated response.
        """
        return self.model.generate_response(hate_speech_text)

    def classify_response(self, hate_speech_text: str, response_text: str) -> str:
        """
        Classifies the response to a hate speech text.

        Args:
            hate_speech_text (str): The original hate speech text.
            response_text (str): The response that was generated.

        Returns:
            str: The classification of the response.
        """
        return self.model.classify_response(hate_speech_text, response_text)

if __name__ == '__main__':
    env_vars = load_env_vars()
    openai_api_key = env_vars.get("OPENAI_API_KEY")

    # Example usage with ChatGPT
    chatgpt_model = ChatGPTModel(api_key=openai_api_key, language="english")
    handler = HateSpeechHandler(model=chatgpt_model)
    
    # Example with Llama (no API key needed for local models)
    # llama_model = LlamaModel(language="english")
    # handler = HateSpeechHandler(model=llama_model)

    hate_speech = "This is a hateful comment."
    
    # 1. Generate a response
    response = handler.generate_response(hate_speech)
    print(f"Hate Speech: {hate_speech}")
    print(f"Generated Response: {response}")

    # 2. Classify the response
    classification = handler.classify_response(hate_speech, response)
    print(f"Response Classification: {classification}")
