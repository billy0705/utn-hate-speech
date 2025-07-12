import argparse
from src.data_handler import DataHandler
from src.models import ChatGPTModel, LlamaModel, DeepSeekModel, ClaudeModel
from src.hate_speech_handler import load_env_vars

def main():
    parser = argparse.ArgumentParser(description='Hate speech data processing.')
    parser.add_argument('task', choices=['gather', 'annotate'], help='Task to perform: gather LLM responses or annotate them.')
    parser.add_argument('--model', choices=['chatgpt', 'llama', 'deepseek', 'claude'], required=True, help='Model to use.')
    parser.add_argument('--languages', nargs='+', help='List of languages to process.')
    parser.add_argument('--limit', type=int, help='Limit the number of items to process.')
    parser.add_argument('--data_path', default='database_building/data', help='Path to the data directory.')
    parser.add_argument('--env_file', default='.env', help='Path to the .env file.')
    parser.add_argument('--use_batch', action='store_true', help='Use batch APIs when supported.')

    args = parser.parse_args()

    env_vars = load_env_vars(args.env_file)
    openai_api_key = env_vars.get("OPENAI_API_KEY")
    deepseek_api_key = env_vars.get("DEEPSEEK_API_KEY")
    anthropic_api_key = env_vars.get("ANTHROPIC_API_KEY")

    if args.model == 'chatgpt':
        model = ChatGPTModel(api_key=openai_api_key)
        model_name = 'ChatGPT'
    elif args.model == 'deepseek':
        model = DeepSeekModel(api_key=deepseek_api_key)
        model_name = 'DeepSeek'
    elif args.model == 'claude':
        model = ClaudeModel(api_key=anthropic_api_key)
        model_name = 'Claude'
    elif args.model == 'llama':
        model = LlamaModel()
        model_name = 'Llama'
    else:
        raise ValueError(f"Unknown model: {args.model}")

    data_handler = DataHandler(data_path=args.data_path)

    if args.task == 'gather':
        data_handler.gather_llm_responses(
            model,
            model_name,
            limit=args.limit,
            languages=args.languages,
            use_batch=args.use_batch,
        )
    elif args.task == 'annotate':
        data_handler.annotate_responses(
            model,
            model_name,
            limit=args.limit,
            languages=args.languages,
            use_batch=args.use_batch,
        )

if __name__ == '__main__':
    main()
