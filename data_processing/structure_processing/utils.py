import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, normaltest, boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL
import os
import math
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------- 多特征批量绘图与保存 ---------

def plot_missing(df, save_dir=None):
    """字段缺失比例图，可选择保存"""
    miss_rate = df.isnull().mean()
    plt.figure(figsize=(12, 4))
    miss_rate.plot(kind='bar', color='skyblue')
    plt.ylabel('缺失比例')
    plt.title('各字段缺失比例图')
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/missing_rate.png")
    plt.close()

def plot_distribution(df, col, save_dir=None):
    """单个字段的分布密度直方图"""
    plt.figure(figsize=(7, 4))
    try:
        sns.histplot(df[col].dropna(), kde=True, color='steelblue')
    except Exception:
        plt.hist(df[col].dropna(), bins=30, color='steelblue')
    plt.title(f'{col} 分布直方')
    plt.xlabel(col)
    plt.ylabel('频率')
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{col}_hist_density.png")
    plt.close()

def save_multi_histplots(df, cols, save_path, plots_per_page=15):
    """
    批量画直方图
    """
    os.makedirs(save_path, exist_ok=True)
    colors = sns.color_palette("Paired", plots_per_page)
    n_rows = math.ceil(plots_per_page / 5)
    for j in range(0, len(cols), plots_per_page):
        plt.figure(figsize=(18, n_rows * 4))
        for i in range(plots_per_page):
            idx = j + i
            if idx >= len(cols):
                break
            plt.subplot(n_rows, 5, i + 1)
            try:
                sns.histplot(df[cols[idx]].dropna(), kde=True, color=colors[i])
            except Exception:
                plt.hist(df[cols[idx]].dropna(), bins=30, color=colors[i])
            plt.title(cols[idx], fontsize=10)
            plt.xlabel(cols[idx])
            plt.ylabel("频率")
        plt.tight_layout()
        plt.savefig(f"{save_path}/hist_subpage_{j // plots_per_page + 1}.png")
        plt.close()


def plot_box(df, col, save_dir=None):
    """字段箱线图"""
    plt.figure(figsize=(4, 5))
    try:
        sns.boxplot(y=df[col].dropna(), color='orange')
    except Exception:
        plt.boxplot(df[col].dropna())
    plt.title(f'{col} 箱线图')
    if save_dir:
        plt.savefig(f"{save_dir}/{col}_boxplot.png")
    plt.close()

def save_multi_boxplots(df, cols, save_path, plots_per_page=15):
    os.makedirs(save_path, exist_ok=True)
    colors = sns.color_palette("Paired", plots_per_page)
    n_rows = math.ceil(plots_per_page / 5)
    for j in range(0, len(cols), plots_per_page):
        plt.figure(figsize=(18, n_rows * 4))
        for i in range(plots_per_page):
            if j + i >= len(cols):
                break
            plt.subplot(n_rows, 5, i + 1)
            try:
                sns.boxplot(y=df[cols[j+i]].dropna(), color=colors[i])
            except Exception:
                plt.boxplot(df[cols[j+i]].dropna())
            plt.title(cols[j+i], fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{save_path}/boxplot_subpage_{j//plots_per_page+1}.png")
        plt.close()

def normal_judge(series):
    """正态分布判断。N>=20时优先normaltest，否则shapiro"""
    s = series.dropna()
    if len(s) < 3:
        return False
    try:
        if len(s) >= 20:
            p = normaltest(s)[1]
        else:
            p = shapiro(s)[1]
        return p > 0.05
    except Exception:
        return False


def missing_value_handling(df, thresh=0.3, record_list=None, plot_dir=None, exclude_cols=None):
    """
    检测缺失，剔除缺失比例大于thresh的字段（不包括exclude_cols），绘制缺失比例，并返回需要补充的字段。
    记录被剔除字段及原因。
    """
    print('\n= 缺失值剔除与统计 =')
    plot_missing(df, save_dir=plot_dir)
    miss_rate = df.isnull().mean()
    exclude_cols = exclude_cols or []
    cols_to_check = [c for c in df.columns if c not in exclude_cols]
    miss_rate = miss_rate[cols_to_check]

    drop_cols = miss_rate[miss_rate > thresh].index.tolist()
    for col in drop_cols:
        print(f'字段 {col} 被删除（缺失比例 {miss_rate[col]:.2%}）')
        if record_list is not None:
            record_list.append({'字段': col, '操作': '删除', '原因': f'缺失过高({miss_rate[col]:.2%})'})
    df = df.drop(columns=drop_cols)
    to_impute = miss_rate[(miss_rate > 0) & (miss_rate <= thresh)].index.tolist()
    if to_impute:
        print('需要补充缺失的字段:', to_impute)
    else:
        print('无需要补充的字段')
    return df, to_impute


def fill_missing(df, cols, data_type, id_col=None, time_col=None, record_list=None, plot_dir=None):
    """
    填充缺失值
    """
    for col in cols:
        ser = df[col]
        print(f'\n处理字段: {col}')
        # 判断连续/离散
        if ser.dtype.kind in 'iuf':  # 整/浮点
            if data_type == 'time_series' or (data_type == 'panel' and time_col is not None):
                print('用线性插值法（时间序列）')
                if time_col and time_col in df.columns:
                    df = df.sort_values(time_col)
                else:
                    df = df
                filled = ser.interpolate(method='linear', limit_direction='both')
                df[col] = filled
                method = '线性插值'
            else:
                if normal_judge(ser):
                    print('正态分布，均值填充')
                    df[col] = ser.fillna(ser.mean())
                    method = '均值'
                else:
                    print('非正态分布，中位数填充')
                    df[col] = ser.fillna(ser.median())
                    method = '中位数'
        elif ser.dtype == object or ser.dtype.name == 'category':
            print('分类变量，众数填充')
            mode = ser.mode()
            fill_val = mode[0] if len(mode) > 0 else np.nan
            df[col] = ser.fillna(fill_val)
            method = '众数'
        else:
            print("未知类型处理")
            df[col] = ser.fillna(method='ffill')
            method = 'ffill'
        if record_list is not None:
            record_list.append({'字段': col, '操作': '补充缺失', '原因': method})
        if plot_dir:
            plot_distribution(df, col, save_dir=plot_dir)
    return df


def normalization(df, exclude_cols=None, record_list=None):
    """
    数据标准化处理
    - 正态: ZScore
    - 非正态: MinMax
    - 极值/异常: Robust
    - 右偏且值大于1: log/BoxCox
    - 量级差大: 小数定标
    """
    print('\n= 数据标准化 =')
    exclude_cols = [col for col in (exclude_cols or []) if col is not None]
    numeric_cols = df.select_dtypes(include='number').columns.difference(exclude_cols)
    for col in numeric_cols:
        s = df[col]
        print(f'\n字段:{col}')
        if s.count() == 0:
            mad = 1e-6
        else:
            mad = np.mean(np.abs(s - s.mean()))
            if mad == 0:
                mad = 1e-6  # prevent zero division
        if ((np.abs(s - s.median()) / mad > 6).sum()) > 0.01 * len(s):
            scaler = RobustScaler()
            print('数据极值多，采用RobustScaler')
            df[col] = scaler.fit_transform(s.values.reshape(-1, 1))
            method = 'RobustScaler'
        elif normal_judge(s):
            print('正态分布，ZScore标准化')
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(s.values.reshape(-1, 1))
            method = 'Z-Score'
        else:
            # 判断右偏且值大于1
            if s.min() > 1 and (s.skew() > 1):
                print('右偏且值大于1，尝试对数/BoxCox')
                s1 = np.log1p(s)
                if normal_judge(s1):
                    df[col] = s1
                    method = 'log1p'
                else:
                    try:
                        bc, lmbda = boxcox(s.dropna())
                        s1_box = s.copy()
                        s1_box[~s.isna()] = bc
                        df[col] = s1_box
                        method = 'BoxCox'
                    except Exception:
                        print('BoxCox失败，降级为MinMax')
                        scaler = MinMaxScaler()
                        df[col] = scaler.fit_transform(s.values.reshape(-1, 1))
                        method = 'MinMax'
            elif s.max() > 1E4 or (s.abs().max() / (abs(s.min()) if abs(s.min()) > 1E-4 else 1.0)) > 1E4:
                print('量级差距过大，小数定标法')
                k = np.ceil(np.log10(s.abs().max()))
                factor = 10 ** k
                df[col] = s / factor
                method = 'decimal'
            else:
                print('默认MinMax')
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(s.values.reshape(-1, 1))
                method = 'MinMax'
        if record_list is not None:
            record_list.append({'字段': col, '操作': '标准化', '原因': method})
    return df
def normalization_with_final_scaling(df, exclude_cols=None, record_list=None, final_scale=True):
    df = normalization(df, exclude_cols=exclude_cols, record_list=record_list)
    if final_scale:
        # 对所有数值型特征整体MinMax缩放
        numeric_cols = df.select_dtypes(include='number').columns.difference(exclude_cols or [])
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        if record_list is not None:
            for col in numeric_cols:
                record_list.append({'字段': col, '操作': '最终缩放', '原因': '统一归一到[0,1]'})
    return df

def detect_outlier(
        df, data_type, time_col=None, id_col=None, record_list=None, plot_dir=None, exclude_cols=None
):
    print('\n= 异常值检测 =')
    exclude_cols = exclude_cols or []
    numeric_cols = df.select_dtypes(include='number').columns.difference(exclude_cols)
    outlier_marks = pd.DataFrame(False, index=df.index, columns=numeric_cols)

    # ============== 新增模式选择 ==============
    handle_mode = None
    first_choice = None
    print("\n【异常值处理交互设置】")
    while handle_mode not in ('1', '2'):
        print("请选择异常值字段交互模式：")
        print("1. 单独处理（适合字段少，或每列特性差异大）")
        print("2. 全部处理（适合字段多）")
        handle_mode = input("请输入模式编号（1或2），默认1：").strip() or "1"
    print(f"已选择模式：{'单独交互' if handle_mode == '1' else '批量统一处理'}")
    # ========================================

    for idx, col in enumerate(numeric_cols):
        s = df[col].dropna()
        print(f'\n检测字段:{col}')
        if (data_type == 'time_series') or (data_type == 'panel' and time_col is not None):
            try:
                stl = STL(s, seasonal=13)
                res = stl.fit()
                resid = res.resid
                threshold = 3 * resid.std()
                abn = np.abs(resid) > threshold
                outlier_idx = abn[abn].index
                if plot_dir:
                    res.plot()
                    plt.savefig(f"{plot_dir}/{col}_STL.png")
                    plt.close()
            except Exception:
                print('STL失败，降级为IQR')
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                abn = (s < lower) | (s > upper)
                outlier_idx = s[abn].index
        elif len(numeric_cols) > 10:
            iso = IsolationForest(contamination=0.01, random_state=2024)
            yhat = iso.fit_predict(df[numeric_cols].fillna(0))
            abn_idx = df.index[yhat == -1]
            outlier_idx = abn_idx if col in df.columns else []
        else:
            if normal_judge(s):
                mu, sigma = s.mean(), s.std()
                abn = (np.abs(s - mu) > 3 * sigma)
                outlier_idx = s[abn].index
                if plot_dir:
                    plot_distribution(df, col, save_dir=plot_dir)# 直方图
            else:
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                abn = (s < lower) | (s > upper)
                outlier_idx = s[abn].index
                if plot_dir:
                    save_multi_boxplots(df, col, save_dir=plot_dir) # 箱线图

        outlier_marks.loc[outlier_idx, col] = True
        if len(outlier_idx) > 0:
            print(f'发现异常值{len(outlier_idx)}个; 行索引: {list(outlier_idx)}')

            # 处理方式选项增加“保持不变”
            print("请选择异常值处理方式：")
            print("1. 删除（将异常值置为NaN）")
            print("2. 截断（超过阈值的替换为阈值）")
            print("3. 替换为均值")
            print("4. 保持原样")
            print("默认：1")

            # 仅第一次进入时询问，后续可自动套用
            if handle_mode == "2":
                # 批量统一
                if first_choice is None:
                    choice = input(f"{col}处理选项:").strip() or "1"
                    first_choice = choice
                else:
                    choice = first_choice
            else:
                # 单独
                choice = input(f"{col}处理选项:").strip() or "1"

            if choice == "1":  # 删除
                df.loc[outlier_idx, col] = np.nan
                op = '删除'
            elif choice == "2":  # 截断
                if normal_judge(s):
                    lower, upper = s.mean() - 3 * s.std(), s.mean() + 3 * s.std()
                else:
                    q1, q3 = s.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df.loc[df[col] < lower, col] = lower
                df.loc[df[col] > upper, col] = upper
                op = '截断'
            elif choice == "3":  # 替换为均值
                df.loc[outlier_idx, col] = s.mean()
                op = '替换均值'
            elif choice == "4":  # 保持原样
                # 什么都不做
                op = '保持原样'
            else:
                print('无效输入，默认删除')
                df.loc[outlier_idx, col] = np.nan
                op = '删除'
            if record_list is not None:
                record_list.append({'字段': col, '操作': '异常值处理', '原因': op, '数量': len(outlier_idx)})
    return df

def regularity_check(df, record_list=None, exclude_cols=None):
    exclude_cols = exclude_cols or []
    cols_to_check = [c for c in df.columns if c not in exclude_cols]

    # 1. 检查完整性
    not_full = [c for c in cols_to_check if df[c].isnull().mean() > 0]
    if not_full:
        print("\n字段仍存在缺失，请回到缺失处理步骤:", not_full)
        return df, not_full

    # 2. 找“高基数冗余”和“高稀疏度”字段
    high_card_cols = [col for col in cols_to_check if df[col].nunique() > 0.9 * len(df)]
    sparse_cols = [col for col in cols_to_check if ((df[col] != 0).sum() / len(df)) < .01]

    # 3. 打印出来供用户选择
    print("-" * 40)
    print("【高基数冗余字段】")
    if high_card_cols:
        for col in high_card_cols:
            print(f"{col}\t唯一值数量: {df[col].nunique()}（占比: {df[col].nunique()/len(df):.2%}）")
    else:
        print("无高基数字段")

    print("\n【高稀疏度字段】")
    if sparse_cols:
        for col in sparse_cols:
            nonzero_ratio = (df[col] != 0).sum() / len(df)
            print(f"{col}\t非零占比: {nonzero_ratio:.2%}")
    else:
        print("无高稀疏度字段")
    print("-" * 40)

    # 4. 用户交互：“你想保留哪些？”
    to_review = list(set(high_card_cols + sparse_cols))
    protected_cols = []

    if to_review:
        tmp = input("如需保留上述字段，请输入需要保留的字段名称，用英文逗号分隔（直接回车则全部删除）：\n")
        if tmp.strip():
            protected_cols = [x.strip() for x in tmp.split(',') if x.strip()]
        print("保留字段为: ", protected_cols)
    else:
        print("当前无异常字段可删除。")
        return df, []

    # 5. 只删除未被保护的异常字段
    to_drop = [col for col in to_review if col not in protected_cols]
    for col in to_drop:
        if col in high_card_cols:
            print(f'字段{col}被删除（高基数冗余）')
            if record_list is not None:
                record_list.append({'字段': col, '操作': '删除', '原因': '高基数冗余'})
        elif col in sparse_cols:
            print(f'字段{col}被删除（高稀疏度）')
            if record_list is not None:
                record_list.append({'字段': col, '操作': '删除', '原因': '高稀疏度'})
    df = df.drop(columns=to_drop)
    return df, []


# END OF UTILS

