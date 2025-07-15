import os
import re
import jieba
import pandas as pd
from collections import Counter

# 停用词
default_stopwords = set("""
的 了 在 是 和 与 也 有 不 对 等 就 为 这 及
及其 还 很 各 各自 之 由 所以 但 但却 只 从 其中
可以 到 被 能 并 及 通过 因而 而 由于 而且 如果 但 是 否 则
""".strip().split())

def load_stopwords(stopwords_path=None):
    if stopwords_path and os.path.exists(stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            user_set = set(line.strip() for line in f if line.strip())
            return default_stopwords | user_set
    else:
        return default_stopwords

def clean_and_count(input_folder, output_folder, stopwords_path=None):
    stopwords = load_stopwords(stopwords_path)
    results = {}
    for fname in os.listdir(input_folder):
        if fname.endswith('.txt'):
            path = os.path.join(input_folder, fname)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z]')
            clean_text = pattern.sub(' ', text)
            words = jieba.lcut(clean_text)
            valid = [w for w in words if w.strip() and not w.isdigit() and len(w) >= 2 and w not in stopwords]
            counter = Counter(valid)
            df = pd.DataFrame(counter.items(), columns=['词语', '词频']).sort_values(by='词频', ascending=False)
            out_path = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_词频.xlsx")
            df.to_excel(out_path, index=False)
            results[os.path.splitext(fname)[0]] = df
    return results

def count_with_dict(input_folder, output_folder, dict_path, stopwords_path=None):
    jieba.load_userdict(dict_path)
    stopwords = load_stopwords(stopwords_path)
    results = {}
    for fname in os.listdir(input_folder):
        if fname.endswith('.txt'):
            path = os.path.join(input_folder, fname)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z]')
            clean_text = pattern.sub(' ', text)
            words = jieba.lcut(clean_text)
            # 用用户词典分词后统计
            valid = [w for w in words if w.strip() and not w.isdigit() and len(w) >= 2 and w not in stopwords]
            counter = Counter(valid)
            df = pd.DataFrame(counter.items(), columns=['词语', '词频']).sort_values(by='词频', ascending=False)
            out_path = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_词频_用户词典.xlsx")
            df.to_excel(out_path, index=False)
            results[os.path.splitext(fname)[0]] = df
    return results