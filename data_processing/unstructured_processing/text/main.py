import os
from text_processing import clean_and_count, count_with_dict
from visualization import visualize_wordfreq
from sentiment_analysis import part_of_speech_tagging, compute_tfidf, sentiment_label
# 参数配置
input_folder = r'C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP-Model\data_loading\categorized_data\unstructured_data\text'
output_folder = r'C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP-Model\data_processing\unstructured_processing\text'
custom_dict_path = os.path.join(output_folder, "user_dict.txt")  # 用户词典，
def parse_word_list(s):
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']')):
        return []
    s = s[1:-1].strip()
    # 统一替换全角逗号和分号为英文逗号
    s = s.replace('，', ',').replace('；', ',').replace(' ', '')  # 去空格
    # 分割
    words = [w for w in s.split(',') if w]
    return words

def create_user_dict_v2(dict_path):
    print("请输入自定义词典，例如：[大模型,人工智能,自然语言处理]")
    user_input = input("输入词典内容: ").strip()
    words = parse_word_list(user_input)
    if not words:
        print("未正确获取到词，取消自定义词典创建。")
        return False
    # 保存到文件，每行一个
    with open(dict_path, 'w', encoding='utf-8') as f:
        for w in words:
            f.write(w + '\n')
    print(f"已保存用户词典到: {dict_path}\n词典内容如下：")
    print("\n".join(words))
    return True

# 主函数中替换create_user_dict为create_user_dict_v2即可
def main():
    print("请选择词频统计方式：1 全部分词  2 基于用户词典")
    choice = input("输入1或2: ").strip()
    if choice == '2':
        created = create_user_dict_v2(custom_dict_path)
        if not created:
            print("未创建用户词典，退出。")
            return
        print("加载用户词典统计…")
        freq_results = count_with_dict(input_folder, output_folder, custom_dict_path)
    elif choice == '1':
        print("常规分词词频统计…")
        freq_results = clean_and_count(input_folder, output_folder)
    else:
        print("输入无效，请重新运行并输入1或2")
        return
    # *** 词云可视化 ***
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