import os
import re

import pandas as pd

def load_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.csv')]
    if len(files) < 2:
        raise Exception("目录下少于两个数据文件")
    dfs = []
    for f in files:
        full = os.path.join(folder_path, f)
        if f.endswith('.xlsx'):
            df = pd.read_excel(full)
        else:
            df = pd.read_csv(full, encoding='utf-8', engine='python')
        dfs.append((f, df))
    return dfs

folder = r'C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP Model\data_conversion\data_merge\merge_datasets'
dfs = load_files(folder)

print("检测到数据文件：")
for idx, (name, df) in enumerate(dfs):
    print(f"{idx}: {name}, 共有{df.shape[0]}行, {df.shape[1]}列")

# 展示全部表共同字段
all_fields = [set(df[1].columns) for df in dfs]
common_fields = set.intersection(*all_fields)
print("\n全部表共同字段：", common_fields)

# 用户选择母表
while True:
    try:
        main_idx = int(input(f"\n请选择作为母表的序号（0~{len(dfs)-1}）: "))
        if main_idx < 0 or main_idx >= len(dfs):
            raise Exception
        break
    except:
        print("输入无效，请输入有效的序号！")
main_name, main_df = dfs[main_idx]

# 合并过程
result = main_df.copy()
used_tables = [main_idx]
stats = []

while len(used_tables) < len(dfs):
    unused = [i for i in range(len(dfs)) if i not in used_tables]
    print("\n剩余待合并表有：")
    for i in unused:
        print(f"{i}: {dfs[i][0]}")
    user_input = input("请输入要合并的表序号（支持1个、多个用逗号，或all/*表示全部剩余表）: ").strip()
    # 处理all/*情况
    if user_input.lower() in ['all', '*']:
        selected_idxs = unused
    else:
        selected_idxs = []
        for val in user_input.split(','):
            val = val.strip()
            if val.isdigit() and int(val) in unused:
                selected_idxs.append(int(val))
        # 若输入无效则提示
        if not selected_idxs:
            print("没有有效的表序号被输入，请重试！")
            continue

    for sub_idx in selected_idxs:
        sub_name, sub_df = dfs[sub_idx]
        c_fields = list(set(result.columns) & set(sub_df.columns))
        if not c_fields:
            print(f"与 {sub_name} 没有共同字段，无法合并。")
            used_tables.append(sub_idx)
            continue
        print(f"\n母表与 {sub_name} 共同字段：{c_fields}")
        cond = input(f"请输入合并条件字段（如A and B、A,B，逗号、and、or或空格分隔都可以）: ")
        merge_cols = [field for field in set(re.findall(r'\b\w+\b', cond)) if field in c_fields]
        if not merge_cols:
            print("未指定有效的共同字段，跳过本表。")
            used_tables.append(sub_idx)
            continue

        # 用户选择要加的新字段
        sub_new = list(set(sub_df.columns) - set(result.columns))
        if sub_new:
            print("可补充的新字段有：", sub_new)
            sel_fields = input("请输入要添加的新字段（逗号分隔，空为全选）: ")
            if sel_fields.strip() == "":
                sel_fields = sub_new
            else:
                sel_fields = [f for f in [x.strip() for x in sel_fields.split(",")] if f in sub_new]
        else:
            print(f"{sub_name}无新字段可补充")
            sel_fields = []

        before = result.shape[0]
        try:
            merged = result.merge(sub_df[merge_cols + sel_fields], on=merge_cols, how='left', suffixes=('', f'_{sub_idx}_sub'))
        except Exception as e:
            print(f"合并{sub_name}时出错：{e}，自动跳过。")
            used_tables.append(sub_idx)
            continue
        after = merged.shape[0]
        missing = merged[sel_fields].isna().sum() if sel_fields else 0

        stats.append({
            "step": len(stats)+1,
            "merge_with": sub_name,
            "before_row": before,
            "after_row": after,
            "missing": dict(zip(sel_fields, missing)) if sel_fields else "无新补字段"
        })

        print(f"合并后母表行数: {after}, 新列缺失数: {missing if sel_fields else '无'}")
        result = merged
        used_tables.append(sub_idx)
    # 一次选择合并完所有未合并，就跳出
    if len(used_tables) == len(dfs):
        break

save_path = os.path.join(folder, f"{main_name.replace('.','_')}_multi_merged.xlsx")
result.to_excel(save_path, index=False)
print(f"\n最终合并结果已保存到: {save_path}")

# 打印统计
print("\n每步合并统计：")
for s in stats:
    print(s)