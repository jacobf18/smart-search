# Import standard packages
import os
import sys
import traceback
import multiprocessing
import numpy as np
from tqdm import tqdm
from langchain_ollama import ChatOllama

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

llm = ChatOllama(
    model="llama3.2",
    temperature=0.5,
    n_gpu_layers=-1,
    n_threads=multiprocessing.cpu_count() - 1,
    verbose=False,
    seed=-1 # Random behavior so will not output the same thing always
)

c = llm.invoke('Hello.')

print(c)