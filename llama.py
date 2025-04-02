# Import standard packages
import os
import sys
import traceback
import multiprocessing
import numpy as np
from tqdm import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import AutoModelForCausalLM, AutoTokenizer

with open('huggingface.txt', 'r') as f:
    token = f.read()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token = token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token = token)