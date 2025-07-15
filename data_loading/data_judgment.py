import os
import shutil
import mimetypes
import pandas as pd

source_folder = r'C:\Users\沐阳\Desktop\数据结构模型\测试数据集'##添加需要分析的文件及其路径##
output_root = 'categorized_data'

structured_types = {
    'cross_sectional': 'cross_sectional_data',
    'time_series': 'time_series_data',
    'panel': 'panel_data'
}
unstructured_types = [
    'text', 'image', 'audio', 'bioinformatics', 'raster', 'other'
]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def classify_unstructured(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    # 生信
    if ext in ['.fasta', '.fastq', '.fna', '.bam', '.vcf']:
        return 'bioinformatics'
    if ext in ['.tif', '.tiff', '.geotiff']:
        return 'raster'
    if ext in ['.wav', '.mp3', '.flac']:
        return 'audio'
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
        return 'image'
    if ext in ['.txt', '.md', '.pdf', '.log']:
        return 'text'
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if mime_type.startswith('text/'):
            return 'text'
        if mime_type.startswith('image/'):
            return 'image'
        if mime_type.startswith('audio/'):
            return 'audio'
    return 'other'


def manual_classify_structured(file_path):
    df = None
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, nrows=1000)  # 读多些方便分析
        elif file_path.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path, nrows=1000)
        elif file_path.lower().endswith('.json'):
            df = pd.read_json(file_path)
        else:
            return None
    except Exception as e:
        print(f"无法读取{file_path}: {e}")
        return None

    print(f"\n文件：{file_path}\n字段名:")
    for i, col in enumerate(df.columns):
        print(f"  {i + 1}. {col}")
    code_col = input('请手动输入“样本编码”字段名（没有可直接回车）：').strip()
    time_col = input('请手动输入“时间”字段名（没有可直接回车）：').strip()

    if code_col == '' and time_col == '':
        print("没有输入任何字段，将视为结构化无法判别")
        return None
    # 字段名校验
    if code_col and code_col not in df.columns:
        print(f"警告：未找到字段：{code_col}，请检查字段拼写。")
        return None
    if time_col and time_col not in df.columns:
        print(f"警告：未找到字段：{time_col}，请检查字段拼写。")
        return None

    if code_col == '' and time_col:
        # 一组样本的时序
        if df[time_col].nunique() > 1:
            return 'time_series'
        else:
            return 'cross_sectional'
    if code_col and time_col == '':
        # 只有样本
        return 'cross_sectional'
    if code_col and time_col:
        n_codes = df[code_col].nunique()
        n_times = df[time_col].nunique()
        if n_codes > 1 and n_times > 1:
            return 'panel'
        elif n_codes == 1 and n_times > 1:
            return 'time_series'
        else:
            return 'cross_sectional'


classified_records = []

# 清理旧的分类文件夹
if os.path.exists(output_root):
    shutil.rmtree(output_root)
ensure_dir(output_root)

for fname in os.listdir(source_folder):
    file_path = os.path.join(source_folder, fname)
    if not os.path.isfile(file_path):
        continue
    ext = os.path.splitext(fname)[1].lower()
    # 结构化候选
    if ext in ['.csv', '.xlsx', '.xls', '.json']:
        struct_type = manual_classify_structured(file_path)
        if struct_type in structured_types:
            subfolder = os.path.join(output_root, 'structure_data', structured_types[struct_type])
            ensure_dir(subfolder)
            shutil.copy2(file_path, os.path.join(subfolder, fname))
            classified_records.append([fname, 'structure_data', structured_types[struct_type]])
            continue
    # 非结构化
    unstruct_type = classify_unstructured(file_path)
    subfolder = os.path.join(output_root, 'unstructured_data', unstruct_type)
    ensure_dir(subfolder)
    shutil.copy2(file_path, os.path.join(subfolder, fname))
    classified_records.append([fname, 'unstructured_data', unstruct_type])

# 删除空的子文件夹
for root, dirs, files in os.walk(output_root, topdown=False):
    if not dirs and not files:
        os.rmdir(root)

result_df = pd.DataFrame(classified_records, columns=['文件名', '一级分类', '二级分类'])
print('\n分类结果：')
print(result_df)
result_df.to_csv(os.path.join(output_root, 'Data_structure_classification.csv'), index=False, encoding='utf-8-sig')

print('\n分类完成，结果已保存到:', output_root)