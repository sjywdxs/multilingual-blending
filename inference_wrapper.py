import httpcore
setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')
from googletrans import Translator
from sentence_transformers import SentenceTransformer
from llm_utils import Get_LLM_Response, Get_Results, path_langauge_handler
from translation_utils import Data_Generator, Text_Blender
import argparse
import nltk
nltk.download('punkt')


TOKENIZERS_PARALLELISM = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--lang_combinations", type=list, default=[])
    parser.add_argument("--path_to_data", type=str, default="dataset/gpt-4o/genetics")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--setting", type=str, default="mix_to_mix")
    parser.add_argument("--open_ai_key", type=str, default="Your OpenAI API HERE")
    parser.add_argument("--PERSPECTIVE_API_KEY", type=str, default="Your Perspective API KEY HERE")
    parser.add_argument("--num_iter", type=int, default=1)
    args = parser.parse_args()

    blender = Text_Blender()
    embedding_model = SentenceTransformer(args.embedding_model)
    translator = Translator()
    prompt_generator = Data_Generator()

    all_path, lang_combinations = path_langauge_handler(args.path_to_data)
    result_generator = Get_Results()

    for path_to_data, language_pool in zip(all_path, lang_combinations):
        GPT_invoker = Get_LLM_Response(args.open_ai_key, args.llm_model)
        
        result_generator.generate(path_to_data, 
                                args.llm_model,
                                GPT_invoker,
                                blender,
                                translator,
                                language_pool,
                                args.setting,
                                args.PERSPECTIVE_API_KEY,
                                num_iter=args.num_iter)