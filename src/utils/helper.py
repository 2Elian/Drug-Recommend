from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer
from collections import Counter
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from src.worker.tool.tokenizer import Tokenizer


def calculate_age(birth_date: str, visit_date: str) -> int:
    birth = datetime.strptime(birth_date, "%Y-%m")
    visit = datetime.strptime(visit_date, "%Y-%m")
    age = visit.year - birth.year
    if (visit.month, visit.day if hasattr(visit, "day") else 1) < (birth.month, 1):
        age -= 1
    return age

def token_count(tokenizer, data):
    fin_text = build_medical_prompt(data)
    return tokenizer.count_tokens(fin_text)
