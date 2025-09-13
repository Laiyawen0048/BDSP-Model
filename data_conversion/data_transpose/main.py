import os
import sys
import re
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

# ============== 配置区（可按需修改） ==============
WORK_DIR = r"C:\Users\沐阳\PycharmProjects\pythonProject3\BDSP-Model\data_conversion\data_transpose"
CITY_SEP_PATTERN = r"[，；、,/\s]+"
VALID_EXTS = {".xlsx", ".xls", ".csv"}
EXCEL_ENGINE = "openpyxl"  # 需要：pip install openpyxl
# ===============================================

def list_data_files(directory: str) -> List[Path]:
    d = Path(directory)
    if not d.exists():
        print(f"目录不存在：{d}")
        sys.exit(1)
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return sorted(files, key=lambda p: p.name)

def read_data(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, engine=EXCEL_ENGINE)
    elif suffix == ".csv":
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"不支持的文件类型：{file_path}")

def prompt_user_choice(files: List[Path]) -> List[Path]:
    print("检测到多个文件，请选择处理方式：")
    for idx, f in enumerate(files, 1):
        print(f"  {idx}. {f.name}")
    print("  A. 处理全部文件")
    print("  Q. 取消并退出")

    while True:
        choice = input("请输入序号 / A / Q：").strip().lower()
        if choice == "a":
            return files
        if choice == "q":
            print("已取消。")
            sys.exit(0)
        if choice.isdigit():
            i = int(choice)
            if 1 <= i <= len(files):
                return [files[i - 1]]
        print("输入无效，请重试。")

def infer_year_columns(df: pd.DataFrame, y_start: int, y_end: int) -> List[str]:
    return [str(y) for y in range(y_start, y_end + 1) if str(y) in df.columns]

def prompt_choose_category_column(df: pd.DataFrame) -> str:
    cols = list(map(str, df.columns))
    print("\n文件的列名如下：")
    for i, c in enumerate(cols, 1):
        print(f"  {i}. {c}")
    print("\n请选择城市/类别字段（用于拆分城市），可直接输入列名或序号：")

    while True:
        s = input("列名或序号：").strip()
        if s.isdigit():
            i = int(s)
            if 1 <= i <= len(cols):
                return cols[i - 1]
        for c in cols:
            if s == c or s.lower().strip() == str(c).lower().strip():
                return c
        print("无效输入，请重试。")

def prompt_year_range(df: pd.DataFrame) -> Tuple[int, int]:
    numeric_like = []
    for c in df.columns:
        cs = str(c).strip()
        if cs.isdigit():
            try:
                y = int(cs)
                if 1900 <= y <= 2100:
                    numeric_like.append(y)
            except Exception:
                pass
    if numeric_like:
        hint_min, hint_max = min(numeric_like), max(numeric_like)
        print(f"\n检测到潜在年份列范围：{hint_min} ~ {hint_max}（仅供参考）")
    else:
        print("\n未检测到明显的年份列，请手动输入。")

    def ask_int(prompt: str) -> int:
        while True:
            s = input(prompt).strip()
            if s.isdigit():
                return int(s)
            print("请输入有效的数字年份。")

    y_start = ask_int("请输入起始年份（如 1999）：")
    y_end = ask_int("请输入结束年份（如 2022）：")
    if y_start > y_end:
        y_start, y_end = y_end, y_start
        print(f"注意：起止年份顺序已自动调整为 {y_start} ~ {y_end}")
    return y_start, y_end

def prompt_output_format() -> str:
    """
    让用户选择输出格式：csv 或 xlsx
    返回 'csv' 或 'xlsx'
    """
    print("\n请选择输出格式：")
    print("  1. CSV（.csv）")
    print("  2. Excel（.xlsx）")
    while True:
        s = input("请输入序号（1 或 2）：").strip()
        if s == "1":
            return "csv"
        if s == "2":
            return "xlsx"
        print("输入无效，请输入 1 或 2。")

def split_cities(city_str: Optional[str], sep_pattern: str = CITY_SEP_PATTERN) -> List[str]:
    if pd.isna(city_str):
        return []
    s = str(city_str).strip()
    if not s:
        return []
    parts = [p.strip() for p in re.split(sep_pattern, s) if p.strip() != ""]
    return parts

def save_result(df: pd.DataFrame, file_path: Path, fmt: str) -> Optional[Path]:
    """
    根据 fmt 保存结果：'csv' 或 'xlsx'
    """
    if fmt == "csv":
        out_path = file_path.with_name(f"{file_path.stem}_output.csv")
        try:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"转换完成，已保存 CSV：{out_path}")
            return out_path
        except Exception as e:
            print(f"保存 CSV 失败：{out_path}，错误：{e}")
            return None
    elif fmt == "xlsx":
        out_path = file_path.with_name(f"{file_path.stem}_output.xlsx")
        try:
            # 使用 openpyxl 引擎写入，并设置 sheet 名称
            with pd.ExcelWriter(out_path, engine=EXCEL_ENGINE) as writer:
                df.to_excel(writer, index=False, sheet_name="result")
            print(f"转换完成，已保存 Excel：{out_path}")
            return out_path
        except Exception as e:
            print(f"保存 Excel 失败：{out_path}，错误：{e}")
            print("若提示缺少 openpyxl，请执行：pip install openpyxl")
            return None
    else:
        print(f"未知输出格式：{fmt}")
        return None

def process_single_file(file_path: Path) -> Optional[Path]:
    print("\n" + "=" * 80)
    print(f"正在处理文件：{file_path.name}")
    print("=" * 80)

    try:
        df = read_data(file_path)
    except Exception as e:
        print(f"读取文件失败：{file_path.name}，错误：{e}")
        return None

    # 展示列名并让用户选择类别字段
    cat_col = prompt_choose_category_column(df)

    # 选择年份起止并确定年份列
    y_start, y_end = prompt_year_range(df)
    year_cols = infer_year_columns(df, y_start, y_end)
    if not year_cols:
        print(f"警告：在 {y_start}~{y_end} 范围内未找到任何年份列，跳过该文件。")
        return None

    missing_years = [str(y) for y in range(y_start, y_end + 1) if str(y) not in df.columns]
    if missing_years:
        print("提示：以下年份列未在数据中找到，将跳过：", ", ".join(missing_years))

    non_year_cols = [col for col in df.columns if str(col) not in year_cols]

    # melt
    melted = df.melt(
        id_vars=non_year_cols,
        value_vars=year_cols,
        var_name="Year",
        value_name="数值"
    )

    # 统一城市原始列
    melted["City_raw"] = melted[cat_col].astype(str).str.strip()

    # 拆分城市
    melted["City_list"] = melted["City_raw"].apply(split_cities)

    # 展开
    rows = []
    for _, row in melted.iterrows():
        city_list = row["City_list"]
        year = row["Year"]
        value = row["数值"]

        if pd.isna(year):
            continue
        try:
            year_int = int(str(year))
        except Exception:
            continue

        if not city_list:
            rows.append({"Year": year_int, "City": row[cat_col], "数值": value})
        else:
            for city in city_list:
                rows.append({"Year": year_int, "City": city, "数值": value})

    result_df = pd.DataFrame(rows)

    # 数值转数值，去除空
    result_df["数值"] = pd.to_numeric(result_df["数值"], errors="coerce")
    result_df = result_df.dropna(subset=["数值"])

    # 选择输出格式并保存
    fmt = prompt_output_format()
    return save_result(result_df, file_path, fmt)

def main():
    files = list_data_files(WORK_DIR)
    if not files:
        print(f"目录中没有找到 Excel/CSV 文件：{WORK_DIR}")
        return

    if len(files) == 1:
        targets = files
        print(f"检测到 1 个文件：{files[0].name}，将直接处理。")
    else:
        targets = prompt_user_choice(files)

    for f in targets:
        process_single_file(f)

    print("\n全部处理完成。")

if __name__ == "__main__":
    main()