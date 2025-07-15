import os
from text_processing import clean_and_count, count_with_dict
from visualization import visualize_wordfreq
from sentiment_analysis import part_of_speech_tagging, compute_tfidf, sentiment_label
# 参数配置
input_folder = r'C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP Model\data_loading\categorized_data\unstructured_data\text'
output_folder = r'C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP Model\data_processing\unstructured_processing\text'
custom_dict_path = os.path.join(output_folder, "user_dict.txt")  # 用户词典，

def main():
    print("请选择词频统计方式：1 全部分词  2 基于用户词典")
    choice = input("输入1或2: ").strip()

    # 1. 清洗与词频统计
    if choice == '2' and os.path.exists(custom_dict_path):
        print("加载用户词典统计…")
        freq_results = count_with_dict(input_folder, output_folder, custom_dict_path)
    else:
        print("常规分词词频统计…")
        freq_results = clean_and_count(input_folder, output_folder)

    # 2. 可视化
    print("开始生成词频可视化…")
    for fname, df in freq_results.items():
        visualize_wordfreq(df, fname, output_folder)

    # 3. 词性分析、TF-IDF、情感分析
    print("开始词性分析、TF-IDF、情感分类…")
    for fname, df in freq_results.items():
        pos_tags = part_of_speech_tagging(df)
        tfidf = compute_tfidf(df, freq_results)
        sentiment = sentiment_label(df)
        # 保存或追加到df
        df['词性'] = pos_tags
        df['TF-IDF'] = tfidf
        df['情感'] = sentiment
        out_path = os.path.join(output_folder, f"{fname}_词频_标注_TFIDF_情感.xlsx")
        df.to_excel(out_path, index=False)
        print(f"{fname} 的结果已保存: {out_path}")

if __name__ == '__main__':
    main()