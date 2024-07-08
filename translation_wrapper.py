import pandas as pd
import joblib 
import os
import copy
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import numpy as np
from scipy.special import softmax
import random
from tqdm import tqdm
import time
from googleapiclient import discovery
import json
from nltk.tokenize import word_tokenize
import goslate
import argostranslate.package
import argostranslate.translate
import httpcore
setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')
from googletrans import Translator
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sentence_transformers import SentenceTransformer, util
import re
import argparse
from llm_utils import Response_Evaluator, Get_LLM_Response, Get_Results, path_langauge_handler
from translation_utils import check_point_save, Data_Generator, Text_Blender, compute_similarity
TOKENIZERS_PARALLELISM = False



"""
Translate Questions into desiginated langauge format
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--lang_combinations", type=list, default=[])
    parser.add_argument("--path_to_question", type=str, default="dataset/mailcious_questions.csv")
    parser.add_argument("--path_to_read", type=str, default="dataset/gpt3-5/num_language")
    parser.add_argument("--sim_thres", type=float, default=0.8)
    parser.add_argument("--iter_thres", type=int, default=30)
    args = parser.parse_args()

    # generate translated questions
    blender = Text_Blender()
    embedding_model = SentenceTransformer(args.embedding_model)
    translator = Translator()
    prompt_generator = Data_Generator()
    np.random.seed(1)

    # if no lang_combinations is given, read from given path
    if len(args.lang_combinations) == 0:
        try:
            _, lang_combinations = path_langauge_handler(args.path_to_read)
        except:
            raise Exception("Loading reference path failed!")

    for language_pool in lang_combinations:
        prompt_generator.generate_question_dataset(
                                            args.path_to_question,
                                            blender = blender, 
                                            language_pool = language_pool, 
                                            translator = translator, 
                                            embedding_model = embedding_model,
                                            sim_thres = args.sim_thres, 
                                            iter_thres = args.iter_thres)