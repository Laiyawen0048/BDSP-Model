import pandas as pd
from utils import (missing_value_handling, fill_missing, normalization,
                   detect_outlier, regularity_check,
                   save_multi_boxplots, save_multi_histplots,
                   plot_box, plot_distribution)

def process_panel(df, id_col=None, time_col=None, exclude_cols=None, save_root='.'):
    print("\n========= 面板数据（panel）预处理 =========")
    record_list = []

    # ------------------- 新增字段选择部分 -------------------
    print("数据字段如下：")
    for idx, col in enumerate(df.columns):
        print(f"{idx + 1}. {col}")
    print("\n请列举**要排除的字段**（用逗号,分隔，可以输入字段名，或直接输入编号如: 1,2,5）：")
    user_input = input("要排除的字段：").strip()
    selected_exclude_cols = []
    if user_input:
        raw_choices = [x.strip() for x in user_input.split(',')]
        for c in raw_choices:
            if c.isdigit():  # 用编号选择
                col_idx = int(c) - 1
                if 0 <= col_idx < len(df.columns):
                    selected_exclude_cols.append(df.columns[col_idx])
            else:
                if c in df.columns:
                    selected_exclude_cols.append(c)
    if not exclude_cols:
        exclude_cols = []
    exclude_cols = list(set(exclude_cols + selected_exclude_cols))
    print(f"\n将排除以下字段参与后续预处理/分析: {exclude_cols}\n")
    # ------------------------------------------------------

    df, to_impute = missing_value_handling(
        df, record_list=record_list, plot_dir=save_root, exclude_cols=exclude_cols
    )
    if to_impute:
        to_impute = [col for col in to_impute if col not in exclude_cols]  # 只填补未被排除的
        if to_impute:
            df = fill_missing(df, to_impute, data_type='panel',
                              id_col=id_col, time_col=time_col, record_list=record_list,
                              plot_dir=save_root)

    df = normalization(df, exclude_cols=list(set([id_col, time_col] + exclude_cols)), record_list=record_list)
    df = detect_outlier(df, data_type='panel', time_col=time_col, id_col=id_col,
                        record_list=record_list, plot_dir=save_root, exclude_cols=exclude_cols)

    while True:
        df, fields_to_fix = regularity_check(df, record_list=record_list, exclude_cols=exclude_cols)
        # 只处理未被排除的字段
        fields_to_fix = [col for col in fields_to_fix if col not in exclude_cols]
        if not fields_to_fix:
            break
        for col in fields_to_fix:
            action = input(f'{col}: 填充(1) or 删除(2): ').strip() or "1"
            if action == "1":
                df = fill_missing(df, [col], 'panel', id_col=id_col, time_col=time_col, record_list=record_list,
                                  plot_dir=save_root)
            else:
                df.drop(columns=[col], inplace=True)
                record_list.append({'字段': col, '操作': '删除', '原因': '手动选择'})

    df.to_csv(f"{save_root}/clean_panel.csv", index=False, encoding='utf-8-sig')
    pd.DataFrame(record_list).to_csv(f"{save_root}/cleaning_log_panel.csv", index=False, encoding='utf-8-sig')
    print(f"完成：{save_root}/clean_panel.csv")
    # ============ 批量画箱线图+直方图 ===============
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c not in exclude_cols]

    save_multi_boxplots(df, numeric_cols, save_path=save_root, plots_per_page=15)
    print(f"最终所有特征的箱线图已保存至: {save_root}")
    save_multi_histplots(df, numeric_cols, save_path=save_root, plots_per_page=15)
    print(f"最终所有特征的直方图已保存至: {save_root}")

    # ============ 所有变量单独画箱线图+直方图 ===============
    for col in numeric_cols:
        plot_box(df, col, save_dir=save_root)
        plot_distribution(df, col, save_dir=save_root)
    print(f"所有特征单独的箱线图和直方图亦已保存至: {save_root}")

    return df, record_list