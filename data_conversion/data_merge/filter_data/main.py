import os
from typing import Optional

import pandas as pd

# --------- 配置与数据读取 ---------
# 数据所在目录
DATA_DIR = r"C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP-Model\data_conversion\data_merge\filter_data"

# 可选：直接在此写死要读取的文件名
DEFAULT_FILENAME = ""  # 例如 "data.xlsx" 或 "data.csv"
def get_script_dir() -> str:
    """
    获取脚本所在目录；在无 __file__ 的环境（如交互式）下回退到当前工作目录。
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

# 尝试读取数据表，支持 CSV 或 Excel
def load_dataframe(directory: str, filename: Optional[str] = None):

    if filename:
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"指定的文件不存在：{path}")
    else:
        # 如果没有指定文件名，尝试目录中的任意一个数据文件（按先后顺序读取 first 找到的 CSV/Excel）
        candidates = [f for f in os.listdir(directory) if f.lower().endswith((".csv", ".xlsx", ".xls"))]
        if not candidates:
            raise FileNotFoundError(f"目录中未发现 CSV/Excel 文件：{directory}")
        # 也可以改为让用户选择具体文件
        path = os.path.join(directory, candidates[0])

    # 读取
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine="openpyxl")
    return df, path

def prompt_yes_no(prompt: str, default_no: bool = True) -> bool:
    """
    简洁的 Y/N 输入确认。默认返回否（default_no=True）。
    """
    raw = input(prompt).strip().lower()
    if raw in ("y", "yes"):
        return True
    if raw in ("n", "no"):
        return False
    # 空输入或不识别按默认
    return not default_no

# --------- 主流程 ---------
def main():
    try:
        df, data_path = load_dataframe(DATA_DIR, DEFAULT_FILENAME)
    except Exception as e:
        print(f"读取数据表时出错：{e}")
        return

    print(f"\n已加载数据文件：{data_path}")
    print("\n数据表字段（全部列名）:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")

    # 用户选择需要筛选的字段列
    print("\n请输入要筛选的字段列名称，多个用逗号分隔，顺序即筛选顺序（例如：CITY, YEAR, PR_TYPE）")
    print("提示：列名需与上方显示一致。")
    selected_cols_input = input("请输入字段列名称（按实际需求输入）：").strip()

    if not selected_cols_input:
        print("未输入字段列名称，退出。")
        return

    # 解析字段列
    input_cols = [s.strip() for s in selected_cols_input.split(",") if s.strip()]
    selected_cols = [c for c in input_cols if c in df.columns]
    invalid_cols = [c for c in input_cols if c not in df.columns]

    if invalid_cols:
        print(f"\n警告：以下列名未在数据中找到，已忽略：{invalid_cols}")
    if not selected_cols:
        print("可用的筛选列为空，请重新运行并输入正确的列名。")
        return

    # 单列时打印唯一值；多列时逐列筛选
    if len(selected_cols) == 1:
        col = selected_cols[0]
        unique_vals = df[col].astype(str).unique()
        unique_vals_sorted = sorted([v for v in unique_vals if pd.notna(v) and v != ""])
        print(f"\n字段 '{col}' 的所有类别值（去重后排序）：")
        for v in unique_vals_sorted:
            print(v)

        # 是否按该列筛选
        if prompt_yes_no(f"\n是否要按列 '{col}' 进行筛选？(Y/N) [默认: N]: ", default_no=True):
            user_filter = input("请输入筛选内容（一个或多个，用逗号分隔）：").strip()
            if user_filter:
                filters = [s.strip() for s in user_filter.split(",") if s.strip() != ""]
                current_df = df[df[col].astype(str).isin(filters)]
            else:
                print("未输入筛选内容，保持原数据。")
                current_df = df.copy()
        else:
            current_df = df.copy()

    else:
        # 多列，按顺序逐列筛选
        current_df = df.copy()
        for idx, col in enumerate(selected_cols, start=1):
            if col not in current_df.columns:
                print(f"列 '{col}' 不存在于当前数据，跳过。")
                continue

            unique_vals = current_df[col].astype(str).unique()
            unique_vals_sorted = sorted([v for v in unique_vals if pd.notna(v) and v != ""])
            print(f"\n步骤 {idx}: 字段 '{col}' 的所有类别值（去重后排序）：")
            for v in unique_vals_sorted:
                print(v)

            # 交互式筛选
            if prompt_yes_no(f"\n是否要按列 '{col}' 进行筛选？(Y/N) [默认: N]: ", default_no=True):
                user_filter = input("请输入筛选内容（一个或多个，用逗号分隔）：").strip()
                if user_filter:
                    filters = [s.strip() for s in user_filter.split(",") if s.strip() != ""]
                    if filters:
                        before_count = len(current_df)
                        current_df = current_df[current_df[col].astype(str).isin(filters)]
                        print(f"已按 '{col}' 筛选：{before_count} -> {len(current_df)} 行。")
                    else:
                        print("未输入有效筛选内容，跳过筛选。")
                else:
                    print("未输入筛选内容，跳过筛选。")
            else:
                print("跳过该列的筛选。")

    # 最终输出
    print("\n最终筛选结果（部分预览）:")
    if not current_df.empty:
        print(current_df.head())
    else:
        print("结果为空。")

    print(f"\n总计行数: {len(current_df)}")

    # 保存交互：先问是否保存
    if prompt_yes_no("是否将最终结果保存？请输入 Y/N [默认: N]: ", default_no=True):
        # 选择保存格式
        fmt = input("请选择保存格式（csv/excel）[默认: csv]: ").strip().lower()
        if fmt not in ("csv", "excel", "xlsx", ""):
            print("输入的格式无效，使用默认 csv。")
            fmt = "csv"

        # 获取脚本所在目录，作为保存路径（与代码同一路径）
        script_dir = get_script_dir()

        if fmt == "" or fmt == "csv":
            out_path = os.path.join(script_dir, "filtered_result.csv")
            current_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"结果已保存到（CSV）：{out_path}")
        else:
            out_path = os.path.join(script_dir, "filtered_result.xlsx")
            try:
                current_df.to_excel(out_path, index=False, engine="openpyxl")
            except Exception as e:
                print("保存 Excel 失败，可能缺少 openpyxl 库。请先安装：pip install openpyxl")
                raise
            print(f"结果已保存到（Excel）：{out_path}")
    else:
        print("未保存结果。")

    print("程序结束。")

if __name__ == "__main__":
    main()