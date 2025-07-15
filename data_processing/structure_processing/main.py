# main.py

import os
import pandas as pd
from process_panel import process_panel
from process_time_series import process_time_series
from process_cross_sectional import process_cross_sectional

# ==== 配置区（请据自己Excel实际字段名填写） ====
PANEL_FIELDS = {'id_col': '面板主键列', 'time_col': '面板时间列'}
TS_FIELDS = {'time_col': '时序时间列'}
CROSS_FIELDS = {'id_col': '截面主键列'}
# 根目录路径
root_path = r'C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP Model\data_loading\categorized_data\structure_data'

types_map = {
    'panel_data': {'func': process_panel, 'fields': PANEL_FIELDS, 'tag': 'panel_result'},
    'time_series_data': {'func': process_time_series, 'fields': TS_FIELDS, 'tag': 'time_series_result'},
    'cross_sectional_data': {'func': process_cross_sectional, 'fields': CROSS_FIELDS, 'tag': 'cross_sectional_result'}
}

for dtype, attr in types_map.items():
    folder = os.path.join(root_path, dtype)
    if not os.path.exists(folder):
        print(f"未找到类型目录: {folder}")
        continue
    for filename in os.listdir(folder):
        if filename.endswith(".xlsx") or filename.endswith(".xls") or filename.endswith(".csv"):
            fpath = os.path.join(folder, filename)
            print(f"\n处理【{dtype}】文件：{fpath}")
            # 读取数据
            if filename.endswith('.csv'):
                df = pd.read_csv(fpath)
            else:
                df = pd.read_excel(fpath)
            # 调用对应处理函数
            result_dir = os.path.join(attr['tag'], filename.replace('.xlsx', '').replace('.xls', '').replace('.csv', ''))
            os.makedirs(result_dir, exist_ok=True)
            # 参数字典展开传递
            attr['func'](df,**attr['fields'],save_root=result_dir)