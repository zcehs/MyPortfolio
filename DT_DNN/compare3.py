import os
import csv
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


def plot_method_episode_comparison(
    data_path: str,
    output_path: str,
    methods: List[str],
    user_counts: List[int],
    metrics: List[str],
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

                plt.plot(np.arange(len(data)), data, label=method, alpha=0.85)
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
) -> None:
    """
    每個 metric 畫兩種總覽：
    1. 壓力測試折線圖：x=users，line=method
    2. 分組長條圖：x=users，bar=method
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
                plt.plot(xs, ys, marker="o", label=method)
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
        # 注意：只有一個 user count 時，如果不固定 x 軸範圍，Matplotlib 會自動把畫面縮到 bar 附近，
        # 即使 width 設得非常小，視覺上仍然會看起來很粗。
        bar_width = 0.08
        bar_gap = 0.02
        offset_step = bar_width + bar_gap
        plotted = False

        for i, method in enumerate(methods):
            offset = (i - (method_count - 1) / 2.0) * offset_step
            label_used = False

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
                )[0]
                label_used = True
                plotted = True

                # 數值標在 bar 上；負值則放在 bar 下方附近
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

            # 固定每個 x category 的可視範圍，避免只有一組資料時 matplotlib 自動放大 bar。
            # 例如只有 users=[100] 時，x 軸固定約在 [-0.5, 0.5]，
            # TD3 / Random 會在 0 附近緊密並排，而且 bar 會真的看起來較細。
            plt.xlim(-0.5, len(user_counts) - 0.5)

            plt.legend()
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(bar_dir, f"Bar_{metric}.png"),
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


def compare_all(
    data_path: str,
    methods: Iterable[str],
    user_counts: Iterable[int],
    output_path: Optional[str] = None,
    metrics: Optional[Iterable[str]] = None,
    start_ep: Optional[int] = None,
    end_ep: Optional[int] = None,
    last_n: Optional[int] = 200,
) -> str:
    """
    跑完所有 pressure test / methods 後，由 main2.py 直接呼叫此函式即可。

    Parameters
    ----------
    data_path:
        main2.py 這次 run 的 datadirectory，例如 trainData/0811_142500
    methods:
        config.METHOD，例如 ['TD3', 'Random', 'Greedy']
    user_counts:
        config.MAX_USER，例如 [25, 50, 75, 100, 125]
    output_path:
        預設存到 <data_path>/comparison
    metrics:
        None = 自動掃描所有 metric；也可以手動指定 CORE_METRICS
    start_ep/end_ep:
        若指定，平均值使用 [start_ep:end_ep]
    last_n:
        未指定 start/end 時，預設比較最後 200 episodes
    """
    methods = list(methods)
    user_counts = list(user_counts)

    if output_path is None:
        output_path = os.path.join(data_path, "comparison")
    os.makedirs(output_path, exist_ok=True)

    if metrics is None:
        metrics = discover_metrics(data_path, methods, user_counts)
    else:
        metrics = list(metrics)

    if not metrics:
        print("[compare] 找不到任何可比較的 metric，請確認 *_log 資料夾與檔名。")
        return output_path

    print("\n================ Comparison ================")
    print(f"Data path   : {data_path}")
    print(f"Output path : {output_path}")
    print(f"Methods     : {methods}")
    print(f"Users       : {user_counts}")
    print(f"Metrics     : {metrics}")

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

    # 3. 壓力測試比較圖
    plot_stress_comparison(
        average_table=average_table,
        output_path=output_path,
        methods=methods,
        user_counts=user_counts,
    )

    # 4. CSV summary
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
    # compare.py 單獨執行時的範例。
    # 正常情況建議直接由 main2.py 呼叫 compare_all()。
    try:
        from config import METHOD, MAX_USER

        # 修改成你想手動比較的某一次 trainData run。
        example_data_path = os.path.join("trainData", "0817_121333")
        compare_all(
            data_path=example_data_path,
            methods=METHOD,
            user_counts=MAX_USER,
            last_n=200,
        )
    except Exception as exc:
        print(f"compare.py standalone mode failed: {exc}")
