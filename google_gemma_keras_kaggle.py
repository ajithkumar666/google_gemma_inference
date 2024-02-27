# KAGGLE_USERNAME and KAGGLE_KEY will get from your Kaggle account
# generate from, go to https://www.kaggle.com/settings and click create new token
import datetime
import os
import keras
import keras_nlp
import numpy as np
os.environ["KAGGLE_USERNAME"] = "KAGGLE_USERNAME"
os.environ["KAGGLE_KEY"] = "KAGGLE_KEY"
os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow" or "torch".

def keras_gemma_2b_en():
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
    while True:
        query = input("Enter your prompt: ")
        res = gemma_lm.generate(query, max_length=64)
        print("Query: ",query)
        print("Result: ",res)

keras_gemma_2b_en()
