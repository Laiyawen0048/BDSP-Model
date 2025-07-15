import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

plt.switch_backend('Agg')  # 不弹窗

def visualize_wordfreq(df, fname, output_folder):
    freq_dict = dict(zip(df['词语'], df['词频']))
    wc = WordCloud(font_path='msyh.ttc',
                   background_color='white',
                   width=800, height=600)
    wc.generate_from_frequencies(freq_dict)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    pic_path = os.path.join(output_folder, f"{fname}_词云.png")
    plt.savefig(pic_path, bbox_inches='tight')
    plt.close()
