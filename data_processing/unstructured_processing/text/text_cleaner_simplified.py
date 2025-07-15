import os
import re
import jieba
import pandas as pd
from collections import Counter

# ========== 参数部分 ==========
input_folder = r'C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP Model\data_loading\categorized_data\unstructured_data\text'
output_folder = r'C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP Model\data_processing\unstructured_processing\text'
stopwords_path = None  # 可指定停用词表路径（可选）

# ========== 4. 停用词 ==========
default_stopwords = set("""
的 了 在 是 和 与 也 有 不 对 等 就 为 这 及
及其 还 很 各 各自 之 由 所以 但 但却 只 从 其中
可以 到 被 能 并 及 通过 因而 而 由于 而且 如果 但 是 否 则
""".strip().split())
extra_stopwords = set()
if stopwords_path and os.path.exists(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        extra_stopwords = set(line.strip() for line in f if line.strip())
stopwords = default_stopwords | extra_stopwords

def is_valid_word(w):
    return (w.strip()            # 非空
            and not w.isdigit()  # 不是纯数字
            and len(w) >= 2      # 长度大于等于2
            and w not in stopwords)

for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        input_path = os.path.join(input_folder, filename)

        # 1. 文本读取
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 2. 文本清洗：保留中文和英文（去除标点、数字留空）
        pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z]')
        clean_text = pattern.sub(' ', text)

        # 3. jieba分词
        words = jieba.lcut(clean_text)

        # 4. 过滤：非空、非停用词、非数字，且长度≥2
        words_filtered = [w for w in words if is_valid_word(w)]

        # 5. 词频统计
        counter = Counter(words_filtered)
        df = pd.DataFrame(counter.items(), columns=['词语', '词频']).sort_values(by='词频', ascending=False)

        # 6. 保存结果
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_词频统计.xlsx")
        df.to_excel(output_path, index=False)

        print(f"{filename} 的词频统计已保存到: {output_path}")