# StableDiffuctionBatchRunner.py
# -*- coding: utf-8 -*-

import os
import sys
import csv
import time
import json
import base64
import queue
import random
import threading
import traceback
from datetime import datetime
from io import BytesIO
from collections import deque  # 新增匯入

import requests

# requests 相關的重試匯入
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------------------------
# 配置 / 預設值 (Config / Defaults)
# ---------------------------
# 將固定的目錄路徑改為預設值，實際路徑由 GUI 控制
DEFAULT_PROMPTS_DIR = "prompts"
DEFAULT_OUTPUTS_DIR = "outputs"

# 預設 API 參數
DEFAULT_API = "http://127.0.0.1:7860/sdapi/v1/txt2img"
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1350
DEFAULT_STEPS = 50
DEFAULT_CFG = 8.0
DEFAULT_SAMPLER = "DPM++ 2M Karras"
DEFAULT_DELAY_MS = 1500  # 請求之間的延遲 (毫秒)
RETRY_MAX = 3  # 最大重試次數 (由 requests.Session 處理)
HTTP_TIMEOUT = 86400  # HTTP 請求超時時間 (秒)


# ---------------------------
# 工具函式 (Utilities)
# ---------------------------


def _get_config_path():
    appdata = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
    if appdata:
        return os.path.join(appdata, "stable_batch_runner_config.json")
    return os.path.join(os.path.expanduser("~"), ".stable_batch_runner_config.json")


def load_runner_config():
    """
    讀取設定檔並做路徑正規化（expanduser -> abspath）。
    回傳 dict（若檔案不存在或解析失敗回傳 {}）。
    """
    path = _get_config_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # 若有 prompts_dir / outputs_dir，做 expanduser + abspath
                    for k in ("prompts_dir", "outputs_dir"):
                        v = data.get(k)
                        if isinstance(v, str) and v.strip():
                            try:
                                data[k] = os.path.abspath(os.path.expanduser(v))
                            except Exception:
                                # 若正規化失敗，保留原值（避免拋例外）
                                data[k] = v
                    return data
    except Exception:
        # 寬鬆失敗處理，不讓程式啟動失敗
        pass
    return {}


def save_runner_config(d: dict):
    path = _get_config_path()
    try:
        safe = {}
        if (
            "prompts_dir" in d
            and isinstance(d["prompts_dir"], str)
            and d["prompts_dir"].strip()
        ):
            safe["prompts_dir"] = os.path.abspath(os.path.expanduser(d["prompts_dir"]))
        if (
            "outputs_dir" in d
            and isinstance(d["outputs_dir"], str)
            and d["outputs_dir"].strip()
        ):
            safe["outputs_dir"] = os.path.abspath(os.path.expanduser(d["outputs_dir"]))
        # 可擴充其他欄位，但目前只存 prompts/outputs
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe, f, ensure_ascii=False, indent=2)
    except Exception:
        # 寬鬆失敗處理，避免阻斷 UI
        pass


def ensure_dir(path):
    """確保單一目錄存在。"""
    os.makedirs(path, exist_ok=True)


def safe_prompt_list(path):
    """安全地讀取提示詞檔案，並返回非空行的列表。"""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    # 移除空行
    return [l for l in lines if l]


def get_resume_index(progress_csv_path):
    import re

    if not os.path.exists(progress_csv_path) or os.path.getsize(progress_csv_path) == 0:
        return 0

    try:
        size = os.path.getsize(progress_csv_path)
        # 只讀檔尾，避免大型檔案耗記憶體
        tail_bytes = 8192
        with open(progress_csv_path, "rb") as f:
            if size > tail_bytes:
                f.seek(-tail_bytes, os.SEEK_END)
            data = f.read()

        # Normalize newlines，取最後非空行
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        lines = data.split(b"\n")
        last_line_bytes = None
        for ln in reversed(lines):
            if ln and ln.strip():
                last_line_bytes = ln.strip()
                break

        if not last_line_bytes:
            return 0

        # 嘗試用多種編碼 decode 該行（常見：utf-8, utf-8-sig, Big5, GBK, Shift-JIS, latin-1）
        encodings_try = ["utf-8", "utf-8-sig", "cp950", "cp936", "cp932", "latin-1"]
        last_line = None
        for enc in encodings_try:
            try:
                last_line = last_line_bytes.decode(enc)
                break
            except Exception:
                continue

        if last_line is None:
            # 最後退而求其次，用 replace 解碼
            last_line = last_line_bytes.decode("utf-8", errors="replace")

        # 嘗試用正則抓最前面的整數（有時 CSV 會把 index 包在引號內）
        m = re.search(r'^\s*"?(\d+)"?', last_line)
        if m:
            try:
                return int(m.group(1)) + 1
            except Exception:
                return 0

        # 若正則沒抓到，嘗試以 csv.reader 讀整個檔（搭配多種編碼）
        from collections import deque

        for enc in encodings_try:
            try:
                with open(
                    progress_csv_path, "r", encoding=enc, errors="replace", newline=""
                ) as f:
                    reader = csv.reader(f)
                    last_rows = deque(reader, maxlen=1)
                    if not last_rows:
                        return 0
                    last = last_rows[0]
                    if not last or len(last) == 0:
                        return 0
                    first_cell = str(last[0]).strip()
                    if first_cell == "":
                        return 0
                    if first_cell.lower() == "index":
                        return 0
                    try:
                        last_index = int(first_cell)
                        return last_index + 1
                    except Exception:
                        return 0
            except Exception:
                continue

        # 全部方法都失敗，回傳 0
        return 0

    except Exception as e:
        # 輕量級警示（不要崩潰）
        print(f"警告：讀取 progress.csv 時發生例外，將從頭開始。錯誤: {e}")
        return 0


def append_progress_row(
    progress_csv_path,
    index,
    prompt,
    neg_prompt,
    seed,
    filename,
    steps,
    width,
    height,
    cfg,
    sampler,
    info_json,
):
    import json as _json
    from datetime import datetime

    # 確保目錄存在
    dirname = os.path.dirname(os.path.abspath(progress_csv_path))
    if dirname:
        ensure_dir(dirname)

    header_needed = (
        not os.path.exists(progress_csv_path) or os.path.getsize(progress_csv_path) == 0
    )

    # Helper: 寫入一列（給予 open() 的檔案物件與 writer）
    def _write_row_with_writer(writer, write_header):
        if write_header:
            writer.writerow(
                [
                    "index",
                    "seed",
                    "timestamp",
                    "width",
                    "height",
                    "cfg_scale",
                    "sampler",
                ]
            )
        # 嘗試將數值欄位轉為適當類型或字串以避免錯誤
        try:
            idx_val = int(index)
        except Exception:
            try:
                idx_val = int(float(index))
            except Exception:
                idx_val = index  # 最後退而求其次，保留原值

        # seed 可能為數字或 -1 / -2 或字串，直接保留原樣
        seed_val = seed

        # width / height / cfg 嘗試轉型，失敗則以原值字串存
        try:
            width_val = int(width)
        except Exception:
            width_val = width
        try:
            height_val = int(height)
        except Exception:
            height_val = height
        try:
            cfg_val = float(cfg)
        except Exception:
            cfg_val = cfg

        sampler_val = sampler

        writer.writerow(
            [
                idx_val,
                seed_val,
                datetime.utcnow().isoformat(),
                width_val,
                height_val,
                cfg_val,
                sampler_val,
            ]
        )

    # 主要寫入流程：先嘗試 utf-8-sig，失敗則用 latin-1
    try:
        with open(
            progress_csv_path, "a", encoding="utf-8-sig", errors="replace", newline=""
        ) as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            _write_row_with_writer(writer, header_needed)
    except Exception as e:
        # 記錄錯誤並嘗試 fallback 編碼
        try:
            log_error(
                f"CSV utf-8 寫入失敗，嘗試 fallback 編碼。錯誤: {e}",
                os.path.join(dirname, "errors.log"),
            )
        except Exception:
            # 若 log_error 也失敗，無需中斷（保險處理）
            pass

        try:
            with open(
                progress_csv_path, "a", encoding="latin-1", errors="replace", newline=""
            ) as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                _write_row_with_writer(writer, header_needed)
        except Exception as e2:
            # 最後仍失敗則把錯誤寫進 errors.log（若可能）
            try:
                log_error(
                    f"CSV fallback 寫入也失敗: {e2}",
                    os.path.join(dirname, "errors.log"),
                )
            except Exception:
                pass


def log_error(msg, error_log_path):
    """將錯誤訊息和堆疊追蹤記錄到指定的 errors.log。"""
    # 確保目錄存在
    ensure_dir(os.path.dirname(error_log_path))
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} - {msg}\n")
        f.write(traceback.format_exc() + "\n")


def decode_and_save_image(b64str, filepath):
    """解碼 Base64 字串並將其儲存為 PNG 圖片。"""
    b = base64.b64decode(b64str)
    # 從記憶體中的 BytesIO 物件打開圖片
    img = Image.open(BytesIO(b))
    img.save(filepath)
    return filepath


def sanitize_filename(s: str):
    """清理字串，使其適合作為檔名。"""
    # 移除檔名不允許字元，取前面部分避免太長
    invalid = '<>:"/\\|?*\n\r\t'
    out = "".join(c for c in s if c not in invalid)
    out = out.strip()
    if len(out) > 120:
        out = out[:120]
    return out


# ---------------------------
# 輔助函式：創建帶重試的 requests Session
# ---------------------------
def create_retry_session(max_retries=RETRY_MAX, backoff_factor=1):
    """
    建立一個帶重試策略的 requests.Session，兼容不同 urllib3 版本。
    - 重試：connect/read/status (500/502/503/504)
    - 對 POST 也進行重試（必要時）
    """
    session = requests.Session()

    # 設定重試策略
    retry_kwargs = dict(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        respect_retry_after_header=True,
    )

    # 針對 urllib3 版本差異做 fallback（allowed_methods vs method_whitelist）
    try:
        retry = Retry(**retry_kwargs, allowed_methods=frozenset(["POST"]))
    except TypeError:
        # 舊版 urllib3 可能使用 method_whitelist
        retry = Retry(**retry_kwargs, method_whitelist=frozenset(["POST"]))

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# ---------------------------
# 工作者執行緒 (Worker Thread)
# ---------------------------
class BatchWorker(threading.Thread):
    def __init__(self, prompts, neg_prompts, start_index, gui_queue, config):
        super().__init__(daemon=True)
        self.prompts = prompts
        self.neg_prompts = neg_prompts
        self.index = start_index  # 當前處理的提示詞索引 (0-based)
        self.gui_queue = gui_queue
        self.config = config
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._pause_event.set()  # 啟動時為未暫停狀態 (即運行中)

        # 優化 1: 創建帶重試的 requests Session
        self.session = create_retry_session(RETRY_MAX)

    def pause(self):
        """設定暫停事件，使 run() 中的 wait() 阻塞。"""
        self._pause_event.clear()

    def resume(self):
        """清除暫停事件，使 run() 中的 wait() 繼續。"""
        self._pause_event.set()

    def stop(self):
        """設定停止事件並確保解除暫停，以便執行緒退出。"""
        self._stop_event.set()
        # 確保如果處於暫停狀態，它可以退出
        self._pause_event.set()

    def skip(self):
        """跳過當前正在處理的索引，前進到下一個。"""
        # 增加索引以跳過當前項
        self.index += 1

    def is_stopped(self):
        """檢查停止事件是否已設定。"""
        return self._stop_event.is_set()

    def run(self):
        """執行緒的主要工作迴圈。"""
        total = len(self.prompts)

        # 從 config 獲取路徑
        outputs_dir = self.config["outputs_dir"]
        progress_csv_path = os.path.join(outputs_dir, "progress.csv")
        error_log_path = os.path.join(outputs_dir, "errors.log")

        # 當前索引小於總數且未停止時，繼續
        while self.index < total and not self.is_stopped():
            # 如果已設定暫停，則阻塞
            self._pause_event.wait()

            current_prompt = self.prompts[self.index]
            # 選擇負向提示詞
            neg = ""
            if self.index < len(self.neg_prompts):
                # 選擇與正向提示詞索引匹配的負向提示詞
                neg = self.neg_prompts[self.index]
            elif len(self.neg_prompts) > 0:
                # 如果負向提示詞較少，則使用第一個作為預設 (如果存在)
                neg = self.neg_prompts[0]

            # 發送狀態到 GUI
            self.gui_queue.put(("status", f"正在生成 {self.index+1}/{total}"))
            self.gui_queue.put(("current_prompt", current_prompt))
            self.gui_queue.put(("progress", (self.index, total)))

            # 準備 API 負載 (payload)
            seed = -1
            if self.config["seed_mode"] == "fixed":
                # 使用固定的種子值，如果不是 -1
                seed = (
                    int(self.config.get("seed_value", -1))
                    if self.config.get("seed_value", -1) != -1
                    else -1
                )
            elif self.config["seed_mode"] == "random":
                # 生成隨機種子值
                seed = random.randint(1, 2**31 - 1)

            payload = {
                "prompt": current_prompt,
                "negative_prompt": neg,
                "steps": int(self.config.get("steps", DEFAULT_STEPS)),
                "sampler_name": self.config.get("sampler", DEFAULT_SAMPLER),
                "scheduler": "Karras",
                "cfg_scale": float(self.config.get("cfg_scale", DEFAULT_CFG)),
                "width": int(self.config.get("width", DEFAULT_WIDTH)),
                "height": int(self.config.get("height", DEFAULT_HEIGHT)),
                "seed": seed,
                "n_iter": 1,
                "batch_size": 1,
            }

            # 優化 1: 使用帶重試機制的 Session 進行單次請求
            success = False
            info_json = {}
            used_seed = seed

            try:
                r = self.session.post(
                    self.config["endpoint"], json=payload, timeout=HTTP_TIMEOUT
                )

                # 若狀態碼不是 2xx，raise_for_status() 會拋出 HTTPError
                r.raise_for_status()

                # 嘗試解析 JSON；如果解析失敗，立刻視為處理錯誤
                try:
                    j = r.json()
                except ValueError as e_json:
                    raise RuntimeError(f"無法解析 JSON 回應: {e_json}") from e_json

                # 取得 images 與 info（若無，給預設值）
                images = j.get("images") if isinstance(j, dict) else None
                info_json = j.get("info") if isinstance(j, dict) else {}

                if not images or not isinstance(images, list) or len(images) == 0:
                    raise RuntimeError(
                        f"響應中沒有圖片 (images 欄位不存在或為空)。HTTP {r.status_code} 回應片段: {str(j)[:400]}"
                    )

                # 若成功到這裡，images[0] 應為 base64 圖片字串
                ensure_dir(outputs_dir)

                # 嘗試解析 info 裡的 seed（info 可能是字串或 dict）
                try:
                    if isinstance(info_json, str):
                        parsed_info = json.loads(info_json)
                        if isinstance(parsed_info, dict) and "seed" in parsed_info:
                            used_seed = parsed_info["seed"]
                    elif isinstance(info_json, dict) and "seed" in info_json:
                        used_seed = info_json["seed"]
                except Exception:
                    # 不影響主要流程，使用 payload 或預設 seed
                    pass

                base_name = f"{self.index:05d}_seed{used_seed}"
                safe_name = sanitize_filename(base_name)
                outpath = os.path.join(outputs_dir, f"{safe_name}.png")

                # 儲存圖片（可能在此拋出錯誤）
                decode_and_save_image(images[0], outpath)

                # 記錄進度
                append_progress_row(
                    progress_csv_path,
                    self.index,
                    current_prompt,
                    neg,
                    used_seed,
                    outpath,
                    payload["steps"],
                    payload["width"],
                    payload["height"],
                    payload["cfg_scale"],
                    payload["sampler_name"],
                    info_json,
                )
                self.gui_queue.put(("image_saved", outpath))
                success = True

            except requests.exceptions.RequestException as e:
                msg = f"Index {self.index} 請求最終失敗: {repr(e)}"
                log_error(msg, error_log_path)
                self.gui_queue.put(("log", f"索引 {self.index} 請求最終失敗: {str(e)}"))
                append_progress_row(
                    progress_csv_path,
                    self.index,
                    current_prompt,
                    neg,
                    -1,
                    "",
                    payload["steps"],
                    payload["width"],
                    payload["height"],
                    payload["cfg_scale"],
                    payload["sampler_name"],
                    {"error": str(e)},
                )
            except Exception as e:
                msg = f"Index {self.index} 處理錯誤: {repr(e)}"
                log_error(msg, error_log_path)
                self.gui_queue.put(("log", f"索引 {self.index} 處理錯誤: {str(e)}"))
                append_progress_row(
                    progress_csv_path,
                    self.index,
                    current_prompt,
                    neg,
                    -2,
                    "",
                    payload["steps"],
                    payload["width"],
                    payload["height"],
                    payload["cfg_scale"],
                    payload["sampler_name"],
                    {"error": str(e)},
                )
            # end try/except

            if self.is_stopped():
                break

            if success:
                self.gui_queue.put(("log", f"已儲存索引 {self.index}"))
            else:
                self.gui_queue.put(
                    ("log", f"索引 {self.index} 處理完成，結果失敗或跳過")
                )

            # 前進到下一個索引
            self.index += 1

            # 節流延遲 (throttle delay)
            delay_s = max(
                0.0, float(self.config.get("delay_ms", DEFAULT_DELAY_MS)) / 1000.0
            )
            # 允許在短時間增量睡眠期間停止
            slept = 0.0
            while slept < delay_s and not self.is_stopped():
                time.sleep(min(0.5, delay_s - slept))
                slept += min(0.5, delay_s - slept)

        # 結束或停止
        if self.is_stopped():
            self.gui_queue.put(("status", "已停止"))
        else:
            self.gui_queue.put(("status", "已完成"))


# ---------------------------
# GUI 介面 (GUI)
# ---------------------------
class BatchGUI:
    def __init__(self, root):
        self.root = root
        root.title("批次執行器 - Stable Diffusion (AUTOMATIC1111)")
        root.geometry("920x640")

        # 用於 worker -> gui 通訊的佇列
        self.gui_queue = queue.Queue()

        # 載入提示詞 - 現在只初始化為空列表，由 reload_prompts 實際載入
        self.prompts = []
        self.neg_prompts = []
        self.start_index = 0

        # 配置變數 (Config vars)
        self.prompts_dir_var = tk.StringVar(value=DEFAULT_PROMPTS_DIR)
        self.outputs_dir_var = tk.StringVar(value=DEFAULT_OUTPUTS_DIR)

        # 在 __init__ 建 UI 之後（或建立 StringVar 後）加入：
        def _save_paths_trace(*args):
            try:
                save_runner_config(
                    {
                        "prompts_dir": self.prompts_dir_var.get(),
                        "outputs_dir": self.outputs_dir_var.get(),
                    }
                )
            except Exception:
                pass

        # 只要變動就儲存（write 表示寫入時觸發）
        self.prompts_dir_var.trace_add("write", _save_paths_trace)
        self.outputs_dir_var.trace_add("write", _save_paths_trace)

        # 載入上次儲存的 prompts/outputs（若有）
        # 載入上次儲存的 prompts/outputs（若有） — load_runner_config() 會回傳已正規化的絕對路徑
        try:
            _cfg = load_runner_config()
            if isinstance(_cfg, dict):
                p = _cfg.get("prompts_dir")
                o = _cfg.get("outputs_dir")
                if p:
                    self.prompts_dir_var.set(p)
                if o:
                    self.outputs_dir_var.set(o)
        except Exception:
            # 若讀取失敗，不影響程式啟動
            pass

        self.endpoint_var = tk.StringVar(value=DEFAULT_API)
        self.steps_var = tk.IntVar(value=DEFAULT_STEPS)
        self.width_var = tk.IntVar(value=DEFAULT_WIDTH)
        self.height_var = tk.IntVar(value=DEFAULT_HEIGHT)
        self.cfg_var = tk.DoubleVar(value=DEFAULT_CFG)
        self.sampler_var = tk.StringVar(value=DEFAULT_SAMPLER)
        self.delay_var = tk.IntVar(value=DEFAULT_DELAY_MS)
        self.seed_mode_var = tk.StringVar(value="random")  # "random" 或 "fixed"
        self.seed_value_var = tk.StringVar(value="-1")
        # 手動設定起始索引變數 (顯示 1-based)
        self.manual_start_index_var = tk.StringVar(value="1")

        # 工作者執行緒 (Worker)
        self.worker = None

        self._build_ui()
        # 初始載入提示詞
        self.reload_prompts()

        # 設定視窗關閉時的處理程序
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        # 啟動定期的 GUI 更新處理程序
        self.root.after(200, self._process_gui_queue)

    def _build_ui(self):
        """建立 GUI 介面佈局和元件。"""
        # 左側框架：控制項
        left = ttk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # --- 新增：資料夾路徑設定 ---
        dir_frame = ttk.LabelFrame(left, text="路徑設定 (Path Settings)")
        dir_frame.pack(anchor="w", fill="x", pady=5)

        # Prompts 資料夾
        pf = ttk.Frame(dir_frame)
        pf.pack(anchor="w", fill="x", padx=5, pady=5)
        ttk.Label(pf, text="Prompts 資料夾:").pack(side=tk.LEFT)
        ttk.Entry(pf, textvariable=self.prompts_dir_var).pack(
            side=tk.LEFT, expand=True, fill="x", padx=(4, 0)
        )
        ttk.Button(pf, text="瀏覽...", command=self._browse_prompts_dir).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        # Outputs 資料夾
        of = ttk.Frame(dir_frame)
        of.pack(anchor="w", fill="x", padx=5, pady=5)
        ttk.Label(of, text="輸出資料夾:").pack(side=tk.LEFT)
        ttk.Entry(of, textvariable=self.outputs_dir_var).pack(
            side=tk.LEFT, expand=True, fill="x", padx=(18, 0)
        )
        ttk.Button(of, text="瀏覽...", command=self._browse_outputs_dir).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        # --- 結束：資料夾路徑設定 ---

        # --- API 與參數設定 ---
        param_frame = ttk.LabelFrame(left, text="參數設定 (Parameters)")
        param_frame.pack(anchor="w", fill="x", pady=5)

        ttk.Label(param_frame, text="API 端點:").pack(anchor="w", padx=5)
        ttk.Entry(param_frame, textvariable=self.endpoint_var, width=48).pack(
            anchor="w", pady=2, padx=5, fill="x"
        )

        ttk.Label(param_frame, text="步數 (Steps):").pack(anchor="w", padx=5)
        ttk.Entry(param_frame, textvariable=self.steps_var, width=12).pack(
            anchor="w", pady=2, padx=5
        )

        ttk.Label(param_frame, text="寬度 x 高度:").pack(anchor="w", padx=5)
        whf = ttk.Frame(param_frame)
        whf.pack(anchor="w", pady=2, padx=5)
        ttk.Entry(whf, textvariable=self.width_var, width=8).pack(side=tk.LEFT)
        ttk.Label(whf, text=" x ").pack(side=tk.LEFT)
        ttk.Entry(whf, textvariable=self.height_var, width=8).pack(side=tk.LEFT)

        ttk.Label(param_frame, text="CFG 比例 (Scale):").pack(anchor="w", padx=5)
        ttk.Entry(param_frame, textvariable=self.cfg_var, width=12).pack(
            anchor="w", pady=2, padx=5
        )

        ttk.Label(param_frame, text="取樣器 (Sampler):").pack(anchor="w", padx=5)
        ttk.Entry(param_frame, textvariable=self.sampler_var, width=20).pack(
            anchor="w", pady=2, padx=5
        )

        ttk.Label(param_frame, text="請求間延遲 (毫秒):").pack(anchor="w", padx=5)
        ttk.Entry(param_frame, textvariable=self.delay_var, width=12).pack(
            anchor="w", pady=2, padx=5
        )

        # 手動設定起始索引控制項 (顯示給使用者看的是 1-based)
        ttk.Label(param_frame, text="手動設定起始點 (1 = 第 1 筆):").pack(
            anchor="w", pady=(10, 0), padx=5
        )
        ttk.Entry(param_frame, textvariable=self.manual_start_index_var, width=12).pack(
            anchor="w", pady=2, padx=5
        )
        ttk.Label(param_frame, text="(輸入 1 從頭開始, 忽略 progress.csv)").pack(
            anchor="w", pady=(0, 5), padx=5
        )

        ttk.Label(param_frame, text="種子模式 (Seed Mode):").pack(anchor="w", padx=5)
        seedf = ttk.Frame(param_frame)
        seedf.pack(anchor="w", pady=2, padx=5)
        ttk.Radiobutton(
            seedf, text="隨機 (Random)", variable=self.seed_mode_var, value="random"
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            seedf, text="固定 (Fixed)", variable=self.seed_mode_var, value="fixed"
        ).pack(side=tk.LEFT)
        ttk.Entry(seedf, textvariable=self.seed_value_var, width=12).pack(
            side=tk.LEFT, padx=6
        )
        # --- 結束：API 與參數設定 ---

        # 控制按鈕 (control buttons)
        btnf = ttk.Frame(left)
        btnf.pack(anchor="w", pady=12)
        self.start_btn = ttk.Button(btnf, text="開始 (Start)", command=self.on_start)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.pause_btn = ttk.Button(
            btnf, text="暫停 (Pause)", command=self.on_pause, state=tk.DISABLED
        )
        self.pause_btn.pack(side=tk.LEFT, padx=4)
        self.resume_btn = ttk.Button(
            btnf, text="恢復 (Resume)", command=self.on_resume, state=tk.DISABLED
        )
        self.resume_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = ttk.Button(
            btnf, text="停止 (Stop)", command=self.on_stop, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=4)
        self.skip_btn = ttk.Button(
            btnf, text="跳過 (Skip)", command=self.on_skip, state=tk.DISABLED
        )
        self.skip_btn.pack(side=tk.LEFT, padx=4)

        # 快速操作 (quick actions)
        qf = ttk.Frame(left)
        qf.pack(anchor="w", pady=8)
        ttk.Button(qf, text="重新載入提示詞", command=self.reload_prompts).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(qf, text="打開輸出資料夾", command=self.open_outputs).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(qf, text="顯示進度 CSV", command=self.open_progress_csv).pack(
            side=tk.LEFT, padx=4
        )

        # 狀態區 (status area)
        status_area = ttk.Frame(left)
        status_area.pack(anchor="w", fill="x", pady=(10, 0))

        ttk.Label(status_area, text="狀態:").pack(anchor="w")
        self.status_var = tk.StringVar(value="空閒 (Idle)")
        ttk.Label(status_area, textvariable=self.status_var, foreground="blue").pack(
            anchor="w"
        )

        ttk.Label(status_area, text="當前提示詞:").pack(anchor="w", pady=(8, 0))
        self.current_prompt_var = tk.StringVar(value="")
        prompt_box = tk.Text(status_area, wrap=tk.WORD, height=4, width=40)
        prompt_box.pack(anchor="w", fill="x")
        prompt_box.insert(tk.END, "")
        prompt_box.config(state=tk.DISABLED)
        self._prompt_box = prompt_box

        ttk.Label(status_area, text="日誌 (Log):").pack(anchor="w", pady=(8, 0))
        self.log_box = tk.Text(status_area, wrap=tk.WORD, height=6, width=40)
        self.log_box.pack(anchor="w", fill="x")
        self.log_box.insert(tk.END, "")
        self.log_box.config(state=tk.DISABLED)

        # 右側框架：預覽 + 進度
        right = ttk.Frame(self.root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(right, text="預覽 (Preview)").pack(anchor="w")
        self.preview_label = ttk.Label(right)
        self.preview_label.pack(anchor="center", pady=6)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progressbar = ttk.Progressbar(
            right, variable=self.progress_var, maximum=1.0  # 0.0 到 1.0 的比例
        )
        self.progressbar.pack(fill=tk.X, padx=6, pady=6)
        self.progress_text_var = tk.StringVar(value="0/0")
        ttk.Label(right, textvariable=self.progress_text_var).pack(anchor="center")

    # -------------------------
    # GUI 輔助方法 (GUI helper methods)
    # -------------------------
    def _browse_prompts_dir(self):
        """瀏覽並設定 Prompts 資料夾"""
        path = filedialog.askdirectory(title="選擇 Prompts 資料夾")

        if path:
            self.prompts_dir_var.set(path)
            # 儲存設定（即時記憶）
            try:
                save_runner_config(
                    {
                        "prompts_dir": os.path.abspath(self.prompts_dir_var.get()),
                        "outputs_dir": os.path.abspath(self.outputs_dir_var.get()),
                    }
                )

            except Exception:
                pass
            self.reload_prompts()

    def _browse_outputs_dir(self):
        """瀏覽並設定輸出資料夾"""
        path = filedialog.askdirectory(title="選擇輸出資料夾")

        if path:
            self.outputs_dir_var.set(path)
            # 儲存設定（即時記憶）
            try:
                save_runner_config(
                    {
                        "prompts_dir": os.path.abspath(self.prompts_dir_var.get()),
                        "outputs_dir": os.path.abspath(self.outputs_dir_var.get()),
                    }
                )

            except Exception:
                pass
            self.reload_prompts()

    def _set_prompt_texts(self):
        """設定當前提示詞文字方塊的內容 (基於 1-based 輸入計算 0-based 索引)。"""
        try:
            manual_input = int(self.manual_start_index_var.get())
            current_idx = max(0, manual_input - 1)
        except ValueError:
            current_idx = self.start_index

        self._prompt_box.config(state=tk.NORMAL)
        self._prompt_box.delete("1.0", tk.END)
        if 0 <= current_idx < len(self.prompts):
            self._prompt_box.insert(tk.END, self.prompts[current_idx])
        else:
            self._prompt_box.insert(
                tk.END, f"索引 {current_idx} 超出範圍，總計 {len(self.prompts)} 行。"
            )
        self._prompt_box.config(state=tk.DISABLED)

    def _log(self, s):
        """將訊息添加到日誌文字方塊。"""
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} {s}\n")
        self.log_box.see(tk.END)  # 滾動到最新日誌
        self.log_box.config(state=tk.DISABLED)

    def reload_prompts(self):
        """從指定路徑重新載入提示詞檔案並更新起始索引。"""
        prompts_dir = self.prompts_dir_var.get()
        outputs_dir = self.outputs_dir_var.get()

        prompts_file = os.path.join(prompts_dir, "prompts.txt")
        neg_prompts_file = os.path.join(prompts_dir, "neg_prompts.txt")
        progress_csv_file = os.path.join(outputs_dir, "progress.csv")

        self.prompts = safe_prompt_list(prompts_file)
        self.neg_prompts = safe_prompt_list(neg_prompts_file)
        self.start_index = get_resume_index(progress_csv_file)

        # 使用恢復的索引來更新手動輸入框 (1-based)
        self.manual_start_index_var.set(str(self.start_index + 1))
        self._set_prompt_texts()
        self._log(f"已從 '{prompts_dir}' 載入提示詞: {len(self.prompts)} 行。")
        self._log(
            f"從 '{progress_csv_file}' 恢復索引: {self.start_index} (第 {self.start_index + 1} 筆)"
        )

    def open_outputs(self):
        """嘗試打開指定的輸出資料夾。"""
        folder = os.path.abspath(self.outputs_dir_var.get())
        ensure_dir(folder)
        if sys.platform == "win32":
            os.startfile(folder)
        else:
            messagebox.showinfo("打開資料夾", f"輸出資料夾: {folder}")

    def open_progress_csv(self):
        """嘗試打開進度 CSV 檔案。"""
        path = os.path.abspath(os.path.join(self.outputs_dir_var.get(), "progress.csv"))
        if os.path.exists(path):
            if sys.platform == "win32":
                os.startfile(path)
            else:
                messagebox.showinfo("進度 CSV", f"進度 CSV: {path}")
        else:
            messagebox.showinfo(
                "進度 CSV", f"在指定輸出資料夾中找不到 progress.csv\n路徑: {path}"
            )

    # -------------------------
    # 控制處理函式 (control handlers)
    # -------------------------
    def on_start(self):
        """處理 '開始' 按鈕點擊事件，啟動工作者執行緒。"""
        prompts_dir = self.prompts_dir_var.get()
        outputs_dir = self.outputs_dir_var.get()

        # 確保目錄存在
        ensure_dir(prompts_dir)
        ensure_dir(outputs_dir)

        # 重要：先暫存使用者在手動起始欄位的輸入（若有）
        # 如果使用者有手動輸入，我們應優先尊重那個值；如果沒有，則使用從 progress.csv 恢復的值。
        user_manual_raw = self.manual_start_index_var.get().strip()

        # 重新載入提示詞以確保是最新的（reload_prompts 會更新 self.start_index 與 manual_start_index_var）
        self.reload_prompts()

        # 強制讓 Tk 更新界面元素，確保 Entry 顯示被立即刷新
        try:
            self.root.update_idletasks()
        except Exception:
            pass

        # 如果使用者之前有手動輸入（而且不是空字串），我們恢復那個輸入值以避免被 reload 覆寫
        if user_manual_raw != "" and user_manual_raw is not None:
            # 若使用者原本輸入的與 reload 後的值不同，且使用者輸入看起來像數字，則尊重使用者輸入
            try:
                int(user_manual_raw)
                # 恢復使用者的自訂值
                self.manual_start_index_var.set(user_manual_raw)
                # 也更新提示詞文字（方便看到目前會從哪個 prompt 開始）
                self._set_prompt_texts()
            except Exception:
                # 若使用者輸入非數字，則保留 reload 的恢復值（reload_prompts 已設定）
                pass

        # 下面解析手動輸入（1-based），並轉成 0-based 的 self.start_index
        try:
            manual_input = int(self.manual_start_index_var.get())
            if manual_input < 1:
                self.start_index = 0
                self._log("手動輸入小於 1，將從第 1 筆 (索引 0) 開始。")
            else:
                self.start_index = manual_input - 1
                self._log(
                    f"手動設定起始點為第 {manual_input} 筆 (索引 {self.start_index})。"
                )
        except ValueError:
            # 如果輸入無效，則使用從 progress.csv 恢復的索引（reload_prompts 已設定 self.start_index）
            progress_csv_path = os.path.join(outputs_dir, "progress.csv")
            self.start_index = get_resume_index(progress_csv_path)
            self.manual_start_index_var.set(str(self.start_index + 1))
            self._set_prompt_texts()
            self._log(
                f"手動起始索引無效，從 progress.csv 恢復索引: {self.start_index}。"
            )

        if self.start_index >= len(self.prompts):
            messagebox.showwarning(
                "超出範圍",
                f"起始點 {self.start_index + 1} 超出提示詞總數 {len(self.prompts)}",
            )
            return

        # 收集當前配置
        config = {
            "prompts_dir": prompts_dir,  # 新增
            "outputs_dir": outputs_dir,  # 新增
            "endpoint": self.endpoint_var.get().strip(),
            "steps": self.steps_var.get(),
            "width": self.width_var.get(),
            "height": self.height_var.get(),
            "cfg_scale": self.cfg_var.get(),
            "sampler": self.sampler_var.get(),
            "delay_ms": self.delay_var.get(),
            "seed_mode": self.seed_mode_var.get(),
            "seed_value": (
                int(self.seed_value_var.get())
                if self.seed_value_var.get().strip()
                else -1
            ),
        }
        # 按開始時儲存當前設定（防止使用者改了 Entry 但沒有按瀏覽）
        try:
            save_runner_config(
                {
                    "prompts_dir": os.path.abspath(self.prompts_dir_var.get()),
                    "outputs_dir": os.path.abspath(self.outputs_dir_var.get()),
                }
            )

        except Exception:
            pass

        # 創建並啟動工作者
        self.worker = BatchWorker(
            self.prompts, self.neg_prompts, self.start_index, self.gui_queue, config
        )
        self.worker.start()

        # 更新按鈕狀態
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.resume_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.skip_btn.config(state=tk.NORMAL)

        self._log("工作者已啟動。")
        self.status_var.set("運行中 (Running)")

    def on_pause(self):
        """處理 '暫停' 按鈕點擊事件。"""
        if self.worker:
            self.worker.pause()
            self.pause_btn.config(state=tk.DISABLED)
            self.resume_btn.config(state=tk.NORMAL)
            self._log("已暫停。")
            self.status_var.set("已暫停 (Paused)")

    def on_resume(self):
        """處理 '恢復' 按鈕點擊事件。"""
        if self.worker:
            self.worker.resume()
            self.pause_btn.config(state=tk.NORMAL)
            self.resume_btn.config(state=tk.DISABLED)
            self._log("已恢復。")
            self.status_var.set("運行中 (Running)")

    def on_stop(self):
        """處理 '停止' 按鈕點擊事件。"""
        if self.worker:
            self.worker.stop()
            self._log("正在停止工作者...")
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.resume_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.skip_btn.config(state=tk.DISABLED)
            self.status_var.set("正在停止 (Stopping)")

    def on_skip(self):
        """處理 '跳過' 按鈕點擊事件。"""
        if self.worker:
            self.worker.skip()
            self._log("已跳過當前索引。")

    # -------------------------
    # GUI 佇列處理器 (GUI queue processor)
    # -------------------------
    def _process_gui_queue(self):
        """定期檢查並處理來自工作者執行緒的訊息佇列。"""
        try:
            while not self.gui_queue.empty():
                item = self.gui_queue.get_nowait()
                key = item[0]
                val = item[1]

                if key == "status":
                    self.status_var.set(val)
                    if val in ["已完成", "已停止"]:
                        self.start_btn.config(state=tk.NORMAL)
                        self.pause_btn.config(state=tk.DISABLED)
                        self.resume_btn.config(state=tk.DISABLED)
                        self.stop_btn.config(state=tk.DISABLED)
                        self.skip_btn.config(state=tk.DISABLED)

                elif key == "current_prompt":
                    self._prompt_box.config(state=tk.NORMAL)
                    self._prompt_box.delete("1.0", tk.END)
                    self._prompt_box.insert(tk.END, val)
                    self._prompt_box.config(state=tk.DISABLED)

                elif key == "progress":
                    idx, total = val
                    self.progress_text_var.set(f"{idx+1}/{total}")
                    if total > 0:
                        # 計算已完成的百分比 (idx 是 0-based, 表示已完成的數量)
                        frac = (idx + 1) / total
                        self.progress_var.set(frac)

                elif key == "image_saved":
                    outpath = val
                    self._log(f"已儲存: {os.path.basename(outpath)}")
                    try:
                        pil = Image.open(outpath)
                        # 縮放圖片以適應預覽區域
                        pil.thumbnail((512, 512))
                        tkimg = ImageTk.PhotoImage(pil)
                        self.preview_label.config(image=tkimg)
                        self.preview_label.image = tkimg
                    except Exception as e:
                        self._log(f"預覽載入失敗: {e}")

                elif key == "log":
                    self._log(val)

                else:
                    self._log(f"佇列: {item}")

        except Exception as e:
            # 這是 GUI 佇列處理本身的錯誤，應記錄但不能讓 GUI 崩潰
            self._log(f"佇列處理錯誤: {e}")
        finally:
            self.root.after(200, self._process_gui_queue)

    def on_close(self):
        """處理視窗關閉事件，如果工作者正在運行，則先詢問是否停止。"""
        if self.worker and self.worker.is_alive():
            if messagebox.askyesno("退出", "工作者正在運行。是否停止並退出?"):
                self.worker.stop()
                time.sleep(0.5)  # 給執行緒一點時間停止
            else:
                return  # 取消關閉
            # 關閉前儲存設定（寬鬆處理以避免任何錯誤阻止關閉）
        try:
            save_runner_config(
                {
                    "prompts_dir": os.path.abspath(self.prompts_dir_var.get()),
                    "outputs_dir": os.path.abspath(self.outputs_dir_var.get()),
                }
            )

        except Exception:
            pass

        self.root.destroy()


# ---------------------------
# 主程式 (Main)
# ---------------------------
def main():
    """程式入口點。"""
    root = tk.Tk()

    # ===== 自動最大化視窗 =====
    root.state("zoomed")

    app = BatchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
