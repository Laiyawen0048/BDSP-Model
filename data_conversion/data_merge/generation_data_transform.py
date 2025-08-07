import pandas as pd

# 定义数据文件的路径
file_path = r'C:\Users\沐阳\Desktop\数据结构模型\非结构数据\生信数据\iris.data'

# 定义列名
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# 读取数据，并将无标题行数据用NA替代
data = pd.read_csv(file_path, names=column_names, header=None)

# 显示数据的前五行
print("原始数据的前五行:")
print(data.head())

# 将数据保存为 CSV 格式
csv_save_path = r'C:\Users\沐阳\Desktop\数据结构模型\非结构数据\生信数据\iris.csv'
data.to_csv(csv_save_path, index=False)

print(f"数据已成功保存为: {csv_save_path}")