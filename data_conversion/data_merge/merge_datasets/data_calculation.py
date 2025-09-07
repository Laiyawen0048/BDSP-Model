import os
import sys
import pandas as pd

# ---------------- 配置：数据源路径（支持 CSV/Excel） ----------------
# 你给定的数据基础路径（不带后缀也可以，程序会尝试不同后缀）
BASE_PATH_NO_EXT = r"C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP-Model\data_conversion\data_merge\filter_data\filtered_result"

# 若你更偏好固定文件名（如 filtered_result.xlsx），可设置：
PREFERRED_FILENAMES = [
    "filtered_result.xlsx",
    "filtered_result.xls",
    "filtered_result.csv",
]

# ---------------- 工具函数 ----------------
def get_script_dir() -> str:
    """获取脚本所在目录；在交互式环境（无 __file__）下回退到当前工作目录。"""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

def try_load_dataframe():
    """
    按优先级尝试加载数据：
    1) 使用 PREFERRED_FILENAMES（与 BASE_PATH_NO_EXT 同目录）
    2) 使用 BASE_PATH_NO_EXT + 常见后缀
    返回：DataFrame, 实际文件路径
    """
    # 构造候选路径列表
    base_dir = os.path.dirname(BASE_PATH_NO_EXT)
    base_name = os.path.basename(BASE_PATH_NO_EXT)
    candidates = []

    # 1) 优先名
    for name in PREFERRED_FILENAMES:
        candidates.append(os.path.join(base_dir, name))

    # 2) 无后缀基础名 + 常见后缀
    for ext in [".xlsx", ".xls", ".csv"]:
        candidates.append(BASE_PATH_NO_EXT + ext)

    # 去重同时保持顺序
    seen = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen:
            unique_candidates.append(p)
            seen.add(p)

    # 逐个尝试读取
    for path in unique_candidates:
        if os.path.exists(path):
            try:
                if path.lower().endswith(".csv"):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path, engine="openpyxl")
                return df, path
            except Exception as e:
                print(f"尝试读取失败：{path} -> {e}")

    # 如果以上均失败，给出提示并退出
    raise FileNotFoundError(
        "未能找到或读取有效的数据文件。已尝试路径：\n" + "\n".join(unique_candidates)
    )

def prompt_choice(prompt, choices, allow_all=False, default=None):
    """
    简单的输入提示函数：
    - prompt: 提示文本
    - choices: 可选集合（小写匹配）
    - allow_all: 是否允许输入 'all'
    - default: 回车默认值（小写）
    返回标准化为小写的选择字符串
    """
    choices_set = set([c.lower() for c in choices])
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        val = raw.lower()
        if allow_all and val == "all":
            return "all"
        if val in choices_set:
            return val
        print(f"输入无效，请在 {sorted(list(choices_set))}" + (" 或 all" if allow_all else "") + " 中选择。")

def compute_stats(series: pd.Series, ops: list[str]) -> dict:
    """
    对单个数值序列按 ops 计算统计值。
    ops 可包含：'mean', 'sum', 'median', 'mode'
    返回：{指标名: 数值}
    说明：众数可能返回多个值，这里取第一个众数；若没有众数则返回 NaN。
    """
    s = pd.to_numeric(series, errors="coerce")
    result = {}
    if "mean" in ops:
        result["mean"] = s.mean()
    if "sum" in ops:
        result["sum"] = s.sum()
    if "median" in ops:
        result["median"] = s.median()
    if "mode" in ops:
        m = s.mode(dropna=True)
        result["mode"] = m.iloc[0] if len(m) > 0 else float("nan")
    return result

def normalize_ops(user_op: str) -> list[str]:
    """
    将用户输入的运算关键词标准化为内部代码列表。
    用户可能输入中文或英文，支持：
    - 平均数/平均/均值/mean
    - 总和数/总和/求和/sum
    - 中位数/median
    - 众数/mode
    - all -> 四项全选
    """
    mapping = {
        "平均数": "mean", "平均": "mean", "均值": "mean", "mean": "mean",
        "总和数": "sum", "总和": "sum", "求和": "sum", "sum": "sum",
        "中位数": "median", "median": "median",
        "众数": "mode", "mode": "mode",
    }
    if user_op == "all":
        return ["mean", "sum", "median", "mode"]
    # 支持逗号分隔多项，如 "平均数, 总和数"
    tokens = [t.strip().lower() for t in user_op.replace("，", ",").split(",") if t.strip() != ""]
    out = []
    for t in tokens:
        # 反向查找中文键：先尝试直接命中英文，再遍历中文
        if t in mapping:
            out.append(mapping[t])
        else:
            # 尝试中文匹配（小写不影响中文）
            for k, v in mapping.items():
                if t == k.lower():
                    out.append(v)
                    break
    # 去重同时保持顺序
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    # 若为空，默认给出全部
    return uniq if uniq else ["mean", "sum", "median", "mode"]

def save_result(df_out: pd.DataFrame, suffix: str):
    """
    将结果保存到脚本同一路径：
    - 优先保存 CSV：stats_result_<suffix>.csv
    - 同时保存 Excel：stats_result_<suffix>.xlsx（如安装 openpyxl）
    """
    script_dir = get_script_dir()
    base = f"stats_result_{suffix}"
    csv_path = os.path.join(script_dir, base + ".csv")
    xlsx_path = os.path.join(script_dir, base + ".xlsx")

    # 保存 CSV
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存 CSV：{csv_path}")

    # 可选保存 Excel
    try:
        df_out.to_excel(xlsx_path, index=False, engine="openpyxl")
        print(f"结果已保存 Excel：{xlsx_path}")
    except Exception as e:
        print("Excel 保存失败（可选）。如需 Excel，请安装 openpyxl：pip install openpyxl")

# ---------------- 主程序 ----------------
def main():
    # 载入数据
    try:
        df, path = try_load_dataframe()
    except Exception as e:
        print(f"加载数据失败：{e}")
        sys.exit(1)

    print(f"已加载数据文件：{path}")
    print("\n数据表字段（全部列名）:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")

    # 选择数值字段（一个或多个）
    target_cols_raw = input("\n请输入需要计算的字段列名称（可多个，用逗号分隔）：").strip()
    if not target_cols_raw:
        print("未输入字段列名称，程序结束。")
        return
    target_cols = [c.strip() for c in target_cols_raw.replace("，", ",").split(",") if c.strip()]
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        print(f"以下字段不存在，已忽略：{missing}")
    target_cols = [c for c in target_cols if c in df.columns]
    if not target_cols:
        print("有效字段为空，程序结束。")
        return

    # 选择计算方式：平均数、总和数、中位数、众数 或 all
    print("\n请输入计算方式（支持中文或英文，多个用逗号；或输入 all 表示全部）：")
    print("- 平均数/平均/均值/mean")
    print("- 总和数/总和/求和/sum")
    print("- 中位数/median")
    print("- 众数/mode")
    ops_raw = input("你的选择：").strip().lower()
    ops = normalize_ops("all" if ops_raw == "all" else ops_raw)
    if not ops:
        print("未选择有效的计算方式，默认全部。")
        ops = ["mean", "sum", "median", "mode"]

    # 是否按组筛选分别计算
    group_choice = input("\n是否按组筛选计算字段列？输入 Y/N [默认: N]: ").strip().lower()
    group_cols = []
    if group_choice in ("y", "yes"):
        grp_raw = input("请输入分组字段（一个或多个，用逗号分隔，可为时间或类别字段）：").strip()
        if grp_raw:
            group_cols = [g.strip() for g in grp_raw.replace("，", ",").split(",") if g.strip()]
            missing_grp = [g for g in group_cols if g not in df.columns]
            if missing_grp:
                print(f"以下分组字段不存在，已忽略：{missing_grp}")
            group_cols = [g for g in group_cols if g in df.columns]
            if not group_cols:
                print("有效分组字段为空，将对全表进行统计。")

    # 计算
    results = []
    if group_cols:
        # 分组后，对每个数值字段分别计算
        grouped = df.groupby(group_cols, dropna=False)
        for keys, subdf in grouped:
            # keys 可能是单值或元组，将其标准化为元组便于拼接
            if not isinstance(keys, tuple):
                keys = (keys,)
            base_row = {gc: val for gc, val in zip(group_cols, keys)}
            for col in target_cols:
                stats = compute_stats(subdf[col], ops)
                row = {**base_row, "field": col, **stats}
                results.append(row)
        result_df = pd.DataFrame(results)
        # 排序：先分组字段，再字段名
        sort_cols = group_cols + ["field"]
        result_df = result_df.sort_values(by=sort_cols, kind="stable")
        save_suffix = "grouped"
    else:
        # 不分组，对每个字段全表计算
        for col in target_cols:
            stats = compute_stats(df[col], ops)
            row = {"field": col, **stats}
            results.append(row)
        result_df = pd.DataFrame(results).sort_values(by=["field"], kind="stable")
        save_suffix = "overall"

    # 输出预览并保存
    print("\n统计结果（预览前10行）：")
    print(result_df.head(10))

    save_result(result_df, suffix=save_suffix)

    print("\n完成。")

if __name__ == "__main__":
    main()