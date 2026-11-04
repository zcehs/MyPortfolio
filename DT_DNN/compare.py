import os
import csv
import inspect
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# 這些是目前 main2.py 會輸出的核心指標。
# metrics=None 時不只畫這些，而是會自動掃描 *_log 資料夾內的所有 txt 指標。
CORE_METRICS = [
    "DT_reward",
    "DNN_reward",
    "Average_update_delay_DT",
    "Average_AoI_DT",
    "Average_AoI_by_step_lst",
    "service_downtime",
    "Accuracy",
    "denined_events",
]


def _format_config_value(value) -> str:
    """將 config 參數轉成適合輸出的文字。"""
    if isinstance(value, np.ndarray):
        return repr(value.tolist())
    return repr(value)


def save_config_parameters(output_path: str) -> Optional[str]:
    """
    將目前 config.py 中的全大寫設定參數輸出成 TXT 與 CSV。

    輸出位置：
      <output_path>/config_parameters.txt
      <output_path>/config_parameters.csv
    """
    try:
        import config
    except Exception as exc:
        print(f"[compare] 無法載入 config.py，因此不輸出 config 參數：{exc}")
        return None

    parameters = []
    for name, value in vars(config).items():
        if not name.isupper() or name.startswith("_"):
            continue

        # 排除 module / function / class，只保留真正的設定值。
        if inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value):
            continue
        if callable(value):
            continue

        parameters.append((name, _format_config_value(value)))

    parameters.sort(key=lambda item: item[0])

    txt_path = os.path.join(output_path, "config_parameters.txt")
    csv_path = os.path.join(output_path, "config_parameters.csv")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("================ Config Parameters ================\n")
        for name, value in parameters:
            f.write(f"{name} = {value}\n")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        writer.writerows(parameters)

    print(f"[compare] Config TXT：{txt_path}")
    print(f"[compare] Config CSV：{csv_path}")
    return txt_path


def read_log_file(file_path: str) -> Optional[np.ndarray]:
    """讀取 main2.py save_data() 輸出的逗號分隔 txt。"""
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return None

    try:
        values = [float(x.strip()) for x in content.replace("\n", ",").split(",") if x.strip()]
    except ValueError:
        return None

    if not values:
        return None

    return np.asarray(values, dtype=float)


def get_log_path(data_path: str, method: str, users: int, metric: str) -> str:
    """
    對應目前 main2.py 的命名方式：
      trainData/<run>/TD3_100_log/TD3_Accuracy_100.txt
    """
    return os.path.join(
        data_path,
        f"{method}_{users}_log",
        f"{method}_{metric}_{users}.txt",
    )


def discover_metrics(
    data_path: str,
    methods: Iterable[str],
    user_counts: Iterable[int],
) -> List[str]:
    """從目前 run 的 *_log 資料夾自動找出所有 metric 名稱。"""
    metrics = set()

    for users in user_counts:
        for method in methods:
            folder = os.path.join(data_path, f"{method}_{users}_log")
            if not os.path.isdir(folder):
                continue

            prefix = f"{method}_"
            suffix = f"_{users}.txt"

            for filename in os.listdir(folder):
                if not (filename.startswith(prefix) and filename.endswith(suffix)):
                    continue
                metric = filename[len(prefix):-len(suffix)]
                if metric:
                    metrics.add(metric)

    # 核心指標優先，其餘（例如 update_delay_DT0、AoI0）照字母排序
    ordered = [m for m in CORE_METRICS if m in metrics]
    ordered.extend(sorted(metrics.difference(ordered)))
    return ordered


def select_average_range(
    data: np.ndarray,
    start_ep: Optional[int] = None,
    end_ep: Optional[int] = None,
    last_n: Optional[int] = 200,
) -> np.ndarray:
    """
    取得比較用的 episode 區間。

    優先順序：
    1. 若 start_ep/end_ep 有指定，就使用 [start_ep:end_ep]
    2. 否則使用最後 last_n 個 episode
    """
    if data is None or len(data) == 0:
        return np.asarray([], dtype=float)

    if start_ep is not None or end_ep is not None:
        start = 0 if start_ep is None else max(0, int(start_ep))
        end = len(data) if end_ep is None else min(len(data), int(end_ep))
        if start >= end:
            return np.asarray([], dtype=float)
        return data[start:end]

    if last_n is None or last_n <= 0:
        return data

    return data[-min(int(last_n), len(data)):]


def get_average(
    file_path: str,
    start_ep: Optional[int] = None,
    end_ep: Optional[int] = None,
    last_n: Optional[int] = 200,
) -> Optional[float]:
    data = read_log_file(file_path)
    if data is None:
        return None

    selected = select_average_range(data, start_ep, end_ep, last_n)
    if len(selected) == 0:
        return None

    finite = selected[np.isfinite(selected)]
    if len(finite) == 0:
        return None

    return float(np.mean(finite))


def build_method_color_map(methods: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    """同一個方法在所有壓力下固定使用同一顏色。"""
    cmap = plt.get_cmap("tab10")
    return {method: cmap(i % cmap.N) for i, method in enumerate(methods)}


def build_user_color_map(user_counts: List[int]) -> Dict[int, Tuple[float, float, float, float]]:
    """同一個壓力（users）在 TD3 壓力比較圖中固定使用同一顏色。"""
    cmap = plt.get_cmap("tab10") if len(user_counts) <= 10 else plt.get_cmap("tab20")
    return {users: cmap(i % cmap.N) for i, users in enumerate(user_counts)}


def moving_average(data: np.ndarray, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """簡單移動平均，用來降低曲線震盪。"""
    if data is None or len(data) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    arr = np.asarray(data, dtype=float)
    if len(arr) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    window = max(1, min(int(window), len(arr)))
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(arr, kernel, mode="valid")
    xs = np.arange(window - 1, len(arr))
    return xs, smoothed


def plot_method_episode_comparison(
    data_path: str,
    output_path: str,
    methods: List[str],
    user_counts: List[int],
    metrics: List[str],
    method_colors: Dict[str, Tuple[float, float, float, float]],
) -> None:
    """固定壓力（users），比較各 method 隨 episode 的曲線。"""
    root = os.path.join(output_path, "By_User_Episode_Curves")
    os.makedirs(root, exist_ok=True)

    for users in user_counts:
        user_dir = os.path.join(root, f"{users}_users")
        os.makedirs(user_dir, exist_ok=True)

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plotted = False

            for method in methods:
                path = get_log_path(data_path, method, users, metric)
                data = read_log_file(path)
                if data is None:
                    continue

                plt.plot(
                    np.arange(len(data)),
                    data,
                    label=method,
                    color=method_colors.get(method),
                    alpha=0.85,
                )
                plotted = True

            if plotted:
                plt.title(f"{metric} - {users} Users")
                plt.xlabel("Episode")
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(user_dir, f"Comparison_{metric}_{users}_users.png"),
                    dpi=250,
                )

            plt.close()


def build_average_table(
    data_path: str,
    methods: List[str],
    user_counts: List[int],
    metrics: List[str],
    start_ep: Optional[int] = None,
    end_ep: Optional[int] = None,
    last_n: Optional[int] = 200,
) -> Dict[str, Dict[str, Dict[int, Optional[float]]]]:
    """metric -> method -> users -> average。"""
    result: Dict[str, Dict[str, Dict[int, Optional[float]]]] = {}

    for metric in metrics:
        result[metric] = {}
        for method in methods:
            result[metric][method] = {}
            for users in user_counts:
                path = get_log_path(data_path, method, users, metric)
                result[metric][method][users] = get_average(
                    path,
                    start_ep=start_ep,
                    end_ep=end_ep,
                    last_n=last_n,
                )

    return result


def plot_stress_comparison(
    average_table: Dict[str, Dict[str, Dict[int, Optional[float]]]],
    output_path: str,
    methods: List[str],
    user_counts: List[int],
    method_colors: Dict[str, Tuple[float, float, float, float]],
) -> None:
    """
    每個 metric 畫兩種總覽：
    1. 壓力測試折線圖：x=users，line=method
    2. 分組長條圖：x=users，bar=method

    注意：這裡同一個 method 在所有 pressure 下都固定同一顏色。
    """
    stress_line_dir = os.path.join(output_path, "Stress_Test_Lines")
    bar_dir = os.path.join(output_path, "Stress_Test_Bars")
    os.makedirs(stress_line_dir, exist_ok=True)
    os.makedirs(bar_dir, exist_ok=True)

    for metric, metric_data in average_table.items():
        # ---------- stress line ----------
        plt.figure(figsize=(10, 6))
        plotted = False
        for method in methods:
            xs = []
            ys = []
            for users in user_counts:
                value = metric_data.get(method, {}).get(users)
                if value is None:
                    continue
                xs.append(users)
                ys.append(value)

            if xs:
                plt.plot(
                    xs,
                    ys,
                    marker="o",
                    label=method,
                    color=method_colors.get(method),
                )
                plotted = True

        if plotted:
            plt.title(f"Stress Test Comparison - {metric}")
            plt.xlabel("Number of Users")
            plt.ylabel(f"Average {metric}")
            plt.xticks(user_counts)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(stress_line_dir, f"Stress_{metric}.png"),
                dpi=250,
            )
        plt.close()

        # ---------- grouped bar ----------
        plt.figure(figsize=(11, 6))
        x = np.arange(len(user_counts), dtype=float)
        method_count = max(len(methods), 1)

        # bar 的「資料座標寬度」。
        # 只有一個 user count 時，固定 x 軸範圍，避免 bar 視覺上被放大。
        bar_width = 0.08
        bar_gap = 0.02
        offset_step = bar_width + bar_gap
        plotted = False

        for i, method in enumerate(methods):
            offset = (i - (method_count - 1) / 2.0) * offset_step
            label_used = False
            color = method_colors.get(method)

            for j, users in enumerate(user_counts):
                value = metric_data.get(method, {}).get(users)
                if value is None:
                    continue

                bar = plt.bar(
                    x[j] + offset,
                    value,
                    bar_width,
                    label=method if not label_used else None,
                    alpha=0.85,
                    color=color,
                )[0]
                label_used = True
                plotted = True

                if value >= 0:
                    y_text = value + max(abs(value) * 0.01, 1e-9)
                    va = "bottom"
                else:
                    y_text = value - max(abs(value) * 0.01, 1e-9)
                    va = "top"

                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_text,
                    f"{value:.4g}",
                    ha="center",
                    va=va,
                    fontsize=7,
                    rotation=90 if len(methods) >= 4 else 0,
                )

        if plotted:
            plt.title(f"Average Performance - {metric}")
            plt.xlabel("Number of Users")
            plt.ylabel(f"Average {metric}")
            plt.xticks(x, [str(u) for u in user_counts])
            plt.xlim(-0.5, len(user_counts) - 0.5)
            plt.legend()
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(bar_dir, f"Bar_{metric}.png"),
                dpi=250,
            )
        plt.close()


def plot_td3_pressure_episode_comparison(
    data_path: str,
    output_path: str,
    user_counts: List[int],
    metrics: List[str],
    td3_method: str = "TD3",
    smooth_window: int = 20,
) -> None:
    """
    只看 TD3：不同壓力下的 episode 收斂曲線比較（像圖二）。
    會輸出兩種：
    1. 原始 TD3 壓力比較
    2. 原始線 + 平滑線 的 TD3 壓力比較（像圖三）
    """
    user_colors = build_user_color_map(user_counts)

    root_raw = os.path.join(output_path, "TD3_Pressure_Episode_Curves")
    root_smooth = os.path.join(output_path, "TD3_Pressure_Episode_Curves_Smoothed")
    os.makedirs(root_raw, exist_ok=True)
    os.makedirs(root_smooth, exist_ok=True)

    for metric in metrics:
        # -------- 原始多壓力比較 --------
        plt.figure(figsize=(10, 6))
        plotted = False
        for users in user_counts:
            path = get_log_path(data_path, td3_method, users, metric)
            data = read_log_file(path)
            if data is None:
                continue
            color = user_colors.get(users)
            plt.plot(np.arange(len(data)), data, label=f"{users} UEs", color=color, alpha=0.9)
            plotted = True

        if plotted:
            plt.title(f"{td3_method} Pressure Comparison - {metric}")
            plt.xlabel("Episodes")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(root_raw, f"{td3_method}_Pressure_{metric}.png"), dpi=250)
        plt.close()

        # -------- 原始 + 平滑 多壓力比較 --------
        plt.figure(figsize=(10, 6))
        plotted = False
        for users in user_counts:
            path = get_log_path(data_path, td3_method, users, metric)
            data = read_log_file(path)
            if data is None:
                continue

            color = user_colors.get(users)
            xs = np.arange(len(data))
            plt.plot(xs, data, color=color, alpha=0.20, linewidth=1.0)

            smooth_xs, smooth_ys = moving_average(data, window=smooth_window)
            if len(smooth_xs) > 0:
                plt.plot(
                    smooth_xs,
                    smooth_ys,
                    color=color,
                    alpha=0.95,
                    linewidth=2.0,
                    label=f"{users} UEs",
                )
                plotted = True

        if plotted:
            plt.title(f"{td3_method} Pressure Comparison (Smoothed) - {metric}")
            plt.xlabel("Episodes")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(root_smooth, f"{td3_method}_Pressure_Smoothed_{metric}.png"),
                dpi=250,
            )
        plt.close()


def plot_td3_smoothed_by_user(
    data_path: str,
    output_path: str,
    user_counts: List[int],
    metrics: List[str],
    td3_method: str = "TD3",
    smooth_window: int = 20,
) -> None:
    """
    每個 TD3、每個壓力，各自重畫一張：
    外層低透明度 = 原始線
    中間高透明度 = 平滑線
    """
    user_colors = build_user_color_map(user_counts)
    root = os.path.join(output_path, "TD3_Smoothed_By_User")
    os.makedirs(root, exist_ok=True)

    for users in user_counts:
        user_dir = os.path.join(root, f"{users}_users")
        os.makedirs(user_dir, exist_ok=True)
        color = user_colors.get(users)

        for metric in metrics:
            path = get_log_path(data_path, td3_method, users, metric)
            data = read_log_file(path)
            if data is None:
                continue

            plt.figure(figsize=(10, 6))
            xs = np.arange(len(data))
            plt.plot(xs, data, color=color, alpha=0.20, linewidth=1.0, label="Raw")

            smooth_xs, smooth_ys = moving_average(data, window=smooth_window)
            if len(smooth_xs) > 0:
                plt.plot(
                    smooth_xs,
                    smooth_ys,
                    color=color,
                    alpha=0.95,
                    linewidth=2.0,
                    label=f"Moving Avg (window={smooth_window})",
                )

            plt.title(f"{td3_method} - {metric} - {users} Users")
            plt.xlabel("Episodes")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(user_dir, f"{td3_method}_Smoothed_{metric}_{users}_users.png"),
                dpi=250,
            )
            plt.close()


def save_summary_csv(
    average_table: Dict[str, Dict[str, Dict[int, Optional[float]]]],
    output_path: str,
    methods: List[str],
    user_counts: List[int],
) -> str:
    """把所有 method × user × metric 的平均值存成 CSV。"""
    csv_path = os.path.join(output_path, "comparison_summary.csv")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Method", "Users", "Average"])

        for metric, metric_data in average_table.items():
            for method in methods:
                for users in user_counts:
                    value = metric_data.get(method, {}).get(users)
                    writer.writerow([
                        metric,
                        method,
                        users,
                        "" if value is None else value,
                    ])

    return csv_path



def find_latest_train_run(train_root: str = "trainData") -> Optional[str]:
    """找出 trainData 底下最新的一個 run 資料夾。"""
    if not os.path.isdir(train_root):
        return None

    candidates = [
        os.path.join(train_root, name)
        for name in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, name))
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]

def compare_all(
    data_path: str,
    methods: Iterable[str],
    user_counts: Iterable[int],
    output_path: Optional[str] = None,
    metrics: Optional[Iterable[str]] = None,
    start_ep: Optional[int] = None,
    end_ep: Optional[int] = None,
    last_n: Optional[int] = 200,
    smooth_window: int = 20,
    td3_method: str = "TD3",
) -> str:
    """
    跑完所有 pressure test / methods 後，由 main2.py 直接呼叫此函式即可。

    另外新增：
    - config 參數輸出
    - 同 method 固定同色的壓力比較 bar 圖
    - 僅 TD3 的多壓力收斂曲線
    - 僅 TD3 的平滑收斂曲線（單壓力與多壓力）
    """
    methods = list(methods)
    user_counts = list(user_counts)

    if output_path is None:
        output_path = os.path.join(data_path, "comparison")
    os.makedirs(output_path, exist_ok=True)

    # 記錄這次實驗的 config.py 設定，方便之後重現與比較。
    save_config_parameters(output_path)

    if metrics is None:
        metrics = discover_metrics(data_path, methods, user_counts)
    else:
        metrics = list(metrics)

    if not metrics:
        print("[compare] 找不到任何可比較的 metric，請確認 *_log 資料夾與檔名。")
        return output_path

    method_colors = build_method_color_map(methods)

    print("\n================ Comparison ================")
    print(f"Data path    : {data_path}")
    print(f"Output path  : {output_path}")
    print(f"Methods      : {methods}")
    print(f"Users        : {user_counts}")
    print(f"Metrics      : {metrics}")
    print(f"Smooth window: {smooth_window}")

    if start_ep is not None or end_ep is not None:
        print(f"Average range: [{start_ep if start_ep is not None else 0}:{end_ep if end_ep is not None else 'end'}]")
    else:
        print(f"Average range: last {last_n} episodes")

    # 1. 每個壓力下，比較不同方法的 episode 曲線
    plot_method_episode_comparison(
        data_path=data_path,
        output_path=output_path,
        methods=methods,
        user_counts=user_counts,
        metrics=list(metrics),
        method_colors=method_colors,
    )

    # 2. 算收斂尾端平均值
    average_table = build_average_table(
        data_path=data_path,
        methods=methods,
        user_counts=user_counts,
        metrics=list(metrics),
        start_ep=start_ep,
        end_ep=end_ep,
        last_n=last_n,
    )

    # 3. 一般壓力測試比較圖（折線 + bar）
    plot_stress_comparison(
        average_table=average_table,
        output_path=output_path,
        methods=methods,
        user_counts=user_counts,
        method_colors=method_colors,
    )

    # 4. 只針對 TD3 畫不同壓力下的收斂曲線比較（像圖二）
    if td3_method in methods:
        plot_td3_pressure_episode_comparison(
            data_path=data_path,
            output_path=output_path,
            user_counts=user_counts,
            metrics=list(metrics),
            td3_method=td3_method,
            smooth_window=smooth_window,
        )

        # 5. 每個 TD3 / 每個壓力，都畫原始線 + 平滑線
        plot_td3_smoothed_by_user(
            data_path=data_path,
            output_path=output_path,
            user_counts=user_counts,
            metrics=list(metrics),
            td3_method=td3_method,
            smooth_window=smooth_window,
        )
    else:
        print(f"[compare] methods 中找不到 {td3_method}，因此略過 TD3 專屬收斂圖。")

    # 6. CSV summary
    csv_path = save_summary_csv(
        average_table=average_table,
        output_path=output_path,
        methods=methods,
        user_counts=user_counts,
    )

    print(f"[compare] 完成，結果輸出到：{output_path}")
    print(f"[compare] 平均值表：{csv_path}")
    print("============================================\n")
    return output_path


if __name__ == "__main__":
    # 用法：
    #   python compare.py
    #   python compare.py trainData/0816_005552
    try:
        from config import METHOD, MAX_USER

        if len(sys.argv) >= 2:
            data_path = sys.argv[1]
        else:
            data_path = find_latest_train_run("trainData")
            if data_path is None:
                data_path = os.path.join("trainData", "0911_131705")

        print(f"[compare] 使用資料夾：{data_path}")
        compare_all(
            data_path=data_path,
            methods=METHOD,
            user_counts=MAX_USER,
            last_n=200,
            smooth_window=20,
            td3_method="TD3",
        )
    except Exception as exc:
        print(f"compare.py standalone mode failed: {exc}")
