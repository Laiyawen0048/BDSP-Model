import os
import re
import pandas as pd

# ========== 工具函数 ==========

def clean_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理数据表中的无效列：
    - 删除列名为 Unnamed: n 的列（多因 Excel 空表头/合并单元格造成）
    - 将纯空白字符串视为缺失，并删除全空列
    """
    if not isinstance(df.columns, pd.Index):
        df.columns = pd.Index(df.columns)
    # 删除 Unnamed 列
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed:\s*\d+$")]
    # 将空字符串/全空格替换为 NaN
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    # 删除全空列
    df = df.dropna(axis=1, how='all')
    return df


def safe_read_csv(path: str) -> pd.DataFrame:
    """
    读取 CSV，优先 utf-8，失败回退 gbk。
    """
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='gbk')


def parse_merge_cols(user_input: str, common_fields: list[str]) -> list[str]:
    """
    解析用户输入的合并字段：
    - 支持逗号、中文逗号、空格、and、or 等分隔
    - 返回与 common_fields 的交集且保持输入顺序的列表
    """
    # 统一替换中文逗号为英文逗号
    normalized = user_input.replace("，", ",")
    # 用 and/or/空白/逗号 作为分隔
    tokens = re.split(r"\s+|,|and|or", normalized, flags=re.IGNORECASE)
    tokens = [t.strip() for t in tokens if t.strip()]
    # 只保留在共同字段中的列，且去重保持顺序
    cf_set = set(common_fields)
    seen = set()
    cols = []
    for t in tokens:
        if t in cf_set and t not in seen:
            cols.append(t)
            seen.add(t)
    return cols


# ========== 核心逻辑 ==========

def load_files(folder_path: str):
    """
    加载目录下的所有 .xlsx 或 .csv 文件，返回 [(文件名, DataFrame), ...]
    读取后对每个 DataFrame 进行列清理（去除 Unnamed 和全空列）。
    """
    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith('.xlsx') or f.lower().endswith('.csv')
    ]
    if len(files) < 2:
        raise Exception("目录下少于两个数据文件")

    dfs = []
    for f in files:
        full = os.path.join(folder_path, f)
        if f.lower().endswith('.xlsx'):
            df = pd.read_excel(full, engine='openpyxl')
        else:
            df = safe_read_csv(full)
        df = clean_df_columns(df)
        dfs.append((f, df))
    return dfs


def main():
    folder = r'C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP-Model\data_conversion\data_merge\merge_datasets'
    dfs = load_files(folder)

    print("检测到数据文件：")
    for idx, (name, df) in enumerate(dfs):
        print(f"{idx}: {name}, 共有{df.shape[0]}行, {df.shape[1]}列")

    # 展示全部表共同字段
    all_fields = [set(df.columns) for _, df in dfs]
    common_fields = set.intersection(*all_fields) if all_fields else set()
    print("\n全部表共同字段：", common_fields if common_fields else "（无共同字段）")

    # 用户选择母表
    while True:
        try:
            main_idx = int(input(f"\n请选择作为母表的序号（0~{len(dfs)-1}）: "))
            if main_idx < 0 or main_idx >= len(dfs):
                raise Exception
            break
        except Exception:
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

        # 处理 all/* 情况
        if user_input.lower() in ['all', '*']:
            selected_idxs = unused
        else:
            selected_idxs = []
            for val in user_input.replace("，", ",").split(','):
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
            cond = input("请输入合并条件字段（如 A and B、A,B，逗号/空格/and/or 分隔均可）: ")
            merge_cols = parse_merge_cols(cond, c_fields)

            if not merge_cols:
                print("未指定有效的共同字段，跳过本表。")
                used_tables.append(sub_idx)
                continue

            # 用户选择要加的新字段
            sub_new = list(set(sub_df.columns) - set(result.columns))
            if sub_new:
                print("可补充的新字段有：", sub_new)
                sel_fields_raw = input("请输入要添加的新字段（逗号分隔，空为全选）: ").strip()
                if sel_fields_raw == "":
                    sel_fields = sub_new
                else:
                    tokens = [x.strip() for x in sel_fields_raw.replace("，", ",").split(",") if x.strip()]
                    sel_fields = [f for f in tokens if f in sub_new]
                    if not sel_fields:
                        print("未选择有效新字段，本次不补充新字段。")
            else:
                print(f"{sub_name} 无新字段可补充")
                sel_fields = []

            before = result.shape[0]
            try:
                # 仅携带需要的列进入 merge，避免无关列引入
                sub_use = sub_df[merge_cols + sel_fields] if sel_fields else sub_df[merge_cols]
                merged = result.merge(
                    sub_use,
                    on=merge_cols,
                    how='left',
                    suffixes=('', f'_{sub_idx}_sub')
                )
            except Exception as e:
                print(f"合并 {sub_name} 时出错：{e}，自动跳过。")
                used_tables.append(sub_idx)
                continue

            after = merged.shape[0]
            missing = merged[sel_fields].isna().sum().to_dict() if sel_fields else "无新补字段"

            stats.append({
                "step": len(stats) + 1,
                "merge_with": sub_name,
                "before_row": before,
                "after_row": after,
                "missing": missing
            })

            print(f"合并后母表行数: {after}, 新列缺失数: {missing if sel_fields else '无'}")
            result = merged
            used_tables.append(sub_idx)

        # 一次选择合并完所有未合并，就跳出
        if len(used_tables) == len(dfs):
            break

    # 保存前再次清理，确保没有 Unnamed 或全空列遗留
    result = clean_df_columns(result)

    save_path = os.path.join(folder, f"{os.path.splitext(main_name)[0].replace('.', '_')}_multi_merged.xlsx")
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result.to_excel(save_path, index=False)
    print(f"\n最终合并结果已保存到: {save_path}")

    # 打印统计
    print("\n每步合并统计：")
    for s in stats:
        print(s)


if __name__ == "__main__":
    main()