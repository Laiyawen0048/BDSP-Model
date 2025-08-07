from snownlp import SnowNLP
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer

# 情感词典
positive_words = {"美好", "优秀", "快乐", "高兴", "满意", "精彩", "友善", "顺利", "积极", "喜欢", "爱", "棒", "赞",
                  "幸福", "成功", "舒适", "可靠", "强大"}
negative_words = {"差", "失望", "伤心", "讨厌", "糟糕", "悲伤", "抱怨", "痛苦", "失败", "烦恼", "消极", "恶劣", "恐惧",
                  "愤怒", "麻烦"}

def listify_df_words(df):
    return ' '.join(df['词语'].astype(str).tolist())

def part_of_speech_tagging(df):
    return [pseg.lcut(w)[0].flag if pseg.lcut(w) else '' for w in df['词语']]

def compute_tfidf(df, all_freq_results):
    corpus = [listify_df_words(doc_df) for doc_df in all_freq_results.values()]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    word2score = {}
    try:
        idx = list(all_freq_results.values()).index(df)
        feature_names = vectorizer.get_feature_names_out()
        for i, word in enumerate(df['词语']):
            if word in feature_names:
                word2score[word] = tfidf_matrix[idx, feature_names.tolist().index(word)]
            else:
                word2score[word] = 0
        return [word2score[w] for w in df['词语']]
    except Exception:
        return [0]*len(df)

def sentiment_label(df):
    """
    融合型情感分析——词典法+ SnowNLP整体判别
    """
    # 1. 词典+词性判断
    pos_count, neg_count = 0, 0
    for word in df['词语']:
        flags = pseg.lcut(str(word))
        if not flags:
            continue
        flag = flags[0].flag
        if flag.startswith("a") or flag.startswith("d") or flag.startswith("v"):
            if word in positive_words:
                pos_count += 1
            elif word in negative_words:
                neg_count += 1

    # 2. SnowNLP模型分析
    doc_text = ' '.join(df['词语'].astype(str).tolist())
    s = SnowNLP(doc_text)
    score = s.sentiments  # 0~1

    # 3. 融合规则
    if pos_count > neg_count:
        label = '积极'
    elif neg_count > pos_count:
        label = '消极'
    else:
        # 若词典法持平，则SnowNLP辅助判断
        if score <= 0.4:
            label = '消极'
        elif score < 0.6:
            label = '中性'
        else:
            label = '积极'
    # 主模块期望每行一标记
    return [label] * len(df)