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
from collections import deque

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
DEFAULT_PROMPTS_DIR = "prompts"
DEFAULT_OUTPUTS_DIR = "outputs"
DEFAULT_API = "http://127.0.0.1:7860/sdapi/v1/txt2img"
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1350
DEFAULT_STEPS = 50
DEFAULT_CFG = 8.0
DEFAULT_SAMPLER = "DPM++ 2M Karras"
DEFAULT_DELAY_MS = 1500
RETRY_MAX = 3
HTTP_TIMEOUT = 86400  # 保持使用者原設定


# ---------------------------
# 工具函式 (Utilities)
# ---------------------------
class ToolTip:
    """為 Tkinter 元件建立 Tooltip。"""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("tahoma", "8", "normal"),
        )
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None


def _get_config_path():
    appdata = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
    if appdata:
        return os.path.join(appdata, "stable_batch_runner_config.json")
    return os.path.join(os.path.expanduser("~"), ".stable_batch_runner_config.json")


def load_runner_config():
    path = _get_config_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k in ("prompts_dir", "outputs_dir"):
                        v = data.get(k)
                        if isinstance(v, str) and v.strip():
                            try:
                                data[k] = os.path.abspath(os.path.expanduser(v))
                            except Exception:
                                data[k] = v
                    return data
    except Exception:
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_prompt_list(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [l for l in lines if l]


def get_resume_index(progress_csv_path):
    import re

    if not os.path.exists(progress_csv_path) or os.path.getsize(progress_csv_path) == 0:
        return 0

    try:
        size = os.path.getsize(progress_csv_path)
        tail_bytes = 8192
        with open(progress_csv_path, "rb") as f:
            if size > tail_bytes:
                f.seek(-tail_bytes, os.SEEK_END)
            data = f.read()

        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        lines = data.split(b"\n")
        last_line_bytes = None
        for ln in reversed(lines):
            if ln and ln.strip():
                last_line_bytes = ln.strip()
                break

        if not last_line_bytes:
            return 0

        encodings_try = ["utf-8", "utf-8-sig", "cp950", "cp936", "cp932", "latin-1"]
        last_line = None
        for enc in encodings_try:
            try:
                last_line = last_line_bytes.decode(enc)
                break
            except Exception:
                continue

        if last_line is None:
            last_line = last_line_bytes.decode("utf-8", errors="replace")

        m = re.search(r'^\s*"?(\d+)"?', last_line)
        if m:
            try:
                return int(m.group(1)) + 1
            except Exception:
                return 0

        # **優化**：移除此處的 from collections import deque
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
                    if not last:
                        return 0
                    first_cell = str(last[0]).strip()
                    if not first_cell or first_cell.lower() == "index":
                        return 0
                    try:
                        return int(first_cell) + 1
                    except Exception:
                        return 0
            except Exception:
                continue
        return 0
    except Exception as e:
        print(f"警告：讀取 progress.csv 時發生例外，將從頭開始。錯誤: {e}")
        return 0


def _try_cast(value, type_func):
    """輔助函式：嘗試轉換型別，失敗則回傳原值。"""
    try:
        return type_func(value)
    except (ValueError, TypeError):
        return value


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
    dirname = os.path.dirname(os.path.abspath(progress_csv_path))
    if dirname:
        ensure_dir(dirname)

    header_needed = (
        not os.path.exists(progress_csv_path) or os.path.getsize(progress_csv_path) == 0
    )

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

        # **優化**：使用輔助函式簡化轉型
        writer.writerow(
            [
                _try_cast(index, int),
                seed,
                datetime.utcnow().isoformat(),
                _try_cast(width, int),
                _try_cast(height, int),
                _try_cast(cfg, float),
                sampler,
            ]
        )

    try:
        with open(
            progress_csv_path, "a", encoding="utf-8-sig", errors="replace", newline=""
        ) as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            _write_row_with_writer(writer, header_needed)
    except Exception as e:
        try:
            log_error(
                f"CSV utf-8 寫入失敗，嘗試 fallback 編碼。錯誤: {e}",
                os.path.join(dirname, "errors.log"),
            )
        except Exception:
            pass
        try:
            with open(
                progress_csv_path, "a", encoding="latin-1", errors="replace", newline=""
            ) as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                _write_row_with_writer(writer, header_needed)
        except Exception as e2:
            try:
                log_error(
                    f"CSV fallback 寫入也失敗: {e2}",
                    os.path.join(dirname, "errors.log"),
                )
            except Exception:
                pass


def log_error(msg, error_log_path):
    ensure_dir(os.path.dirname(error_log_path))
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} - {msg}\n")
        f.write(traceback.format_exc() + "\n")


def decode_and_save_image(b64str, filepath):
    img = Image.open(BytesIO(base64.b64decode(b64str)))
    img.save(filepath)
    return filepath


def sanitize_filename(s: str):
    invalid = '<>:"/\\|?*\n\r\t'
    out = "".join(c for c in s if c not in invalid).strip()
    return out[:120]


def create_retry_session(max_retries=RETRY_MAX, backoff_factor=1):
    session = requests.Session()
    retry_kwargs = dict(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    try:
        retry = Retry(**retry_kwargs, allowed_methods=frozenset(["POST"]))
    except TypeError:
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
        self.index = start_index
        self.gui_queue = gui_queue
        self.config = config
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._pause_event.set()
        self.session = create_retry_session(RETRY_MAX)

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()

    def skip(self):
        self.index += 1

    def is_stopped(self):
        return self._stop_event.is_set()

    # **優化**：將 Payload 構建邏輯獨立成方法
    def _build_payload(self):
        """根據當前索引和設定構建 API payload。"""
        current_prompt = self.prompts[self.index]
        neg = ""
        if self.index < len(self.neg_prompts):
            neg = self.neg_prompts[self.index]
        elif self.neg_prompts:
            neg = self.neg_prompts[0]

        seed = -1
        if self.config["seed_mode"] == "fixed":
            seed = int(self.config.get("seed_value", -1))
        elif self.config["seed_mode"] == "random":
            seed = random.randint(1, 2**31 - 1)

        return {
            "prompt": current_prompt,
            "negative_prompt": neg,
            "steps": int(self.config.get("steps", DEFAULT_STEPS)),
            "sampler_name": self.config.get("sampler", DEFAULT_SAMPLER),
            "cfg_scale": float(self.config.get("cfg_scale", DEFAULT_CFG)),
            "width": int(self.config.get("width", DEFAULT_WIDTH)),
            "height": int(self.config.get("height", DEFAULT_HEIGHT)),
            "seed": seed,
            "n_iter": 1,
            "batch_size": 1,
        }

    def run(self):
        total = len(self.prompts)
        outputs_dir = self.config["outputs_dir"]
        progress_csv_path = os.path.join(outputs_dir, "progress.csv")
        error_log_path = os.path.join(outputs_dir, "errors.log")

        while self.index < total and not self.is_stopped():
            self._pause_event.wait()

            # **優化**：呼叫獨立的方法來構建 payload
            payload = self._build_payload()
            current_prompt = payload["prompt"]
            neg = payload["negative_prompt"]

            self.gui_queue.put(("status", f"正在生成 {self.index+1}/{total}"))
            self.gui_queue.put(("current_prompt", current_prompt))
            self.gui_queue.put(("progress", (self.index, total)))

            success = False
            info_json = {}
            used_seed = payload["seed"]

            try:
                r = self.session.post(
                    self.config["endpoint"], json=payload, timeout=HTTP_TIMEOUT
                )
                r.raise_for_status()

                try:
                    j = r.json()
                except ValueError as e_json:
                    raise RuntimeError(f"無法解析 JSON 回應: {e_json}") from e_json

                images = j.get("images") if isinstance(j, dict) else None
                info_json = j.get("info") if isinstance(j, dict) else {}

                if not images or not isinstance(images, list) or len(images) == 0:
                    raise RuntimeError(
                        f"響應中沒有圖片。HTTP {r.status_code} 回應: {str(j)[:400]}"
                    )

                ensure_dir(outputs_dir)

                try:
                    if isinstance(info_json, str):
                        parsed_info = json.loads(info_json)
                        if isinstance(parsed_info, dict) and "seed" in parsed_info:
                            used_seed = parsed_info["seed"]
                    elif isinstance(info_json, dict) and "seed" in info_json:
                        used_seed = info_json["seed"]
                except Exception:
                    pass

                base_name = f"{self.index:05d}_seed{used_seed}"
                safe_name = sanitize_filename(base_name)
                outpath = os.path.join(outputs_dir, f"{safe_name}.png")

                decode_and_save_image(images[0], outpath)

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

            if self.is_stopped():
                break

            if success:
                self.gui_queue.put(("log", f"已儲存索引 {self.index}"))
            else:
                self.gui_queue.put(("log", f"索引 {self.index} 處理失敗或跳過"))

            self.index += 1

            delay_s = max(
                0.0, float(self.config.get("delay_ms", DEFAULT_DELAY_MS)) / 1000.0
            )
            slept = 0.0
            while slept < delay_s and not self.is_stopped():
                time.sleep(min(0.5, delay_s - slept))
                slept += 0.5

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
        self.gui_queue = queue.Queue()
        self.prompts = []
        self.neg_prompts = []
        self.start_index = 0

        self.prompts_dir_var = tk.StringVar(value=DEFAULT_PROMPTS_DIR)
        self.outputs_dir_var = tk.StringVar(value=DEFAULT_OUTPUTS_DIR)

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

        self.prompts_dir_var.trace_add("write", _save_paths_trace)
        self.outputs_dir_var.trace_add("write", _save_paths_trace)

        try:
            _cfg = load_runner_config()
            if isinstance(_cfg, dict):
                p, o = _cfg.get("prompts_dir"), _cfg.get("outputs_dir")
                if p:
                    self.prompts_dir_var.set(p)
                if o:
                    self.outputs_dir_var.set(o)
        except Exception:
            pass

        self.endpoint_var = tk.StringVar(value=DEFAULT_API)
        self.steps_var = tk.IntVar(value=DEFAULT_STEPS)
        self.width_var = tk.IntVar(value=DEFAULT_WIDTH)
        self.height_var = tk.IntVar(value=DEFAULT_HEIGHT)
        self.cfg_var = tk.DoubleVar(value=DEFAULT_CFG)
        self.sampler_var = tk.StringVar(value=DEFAULT_SAMPLER)
        self.delay_var = tk.IntVar(value=DEFAULT_DELAY_MS)
        self.seed_mode_var = tk.StringVar(value="random")
        self.seed_value_var = tk.StringVar(value="-1")
        self.manual_start_index_var = tk.StringVar(value="1")
        self.worker = None

        self._build_ui()
        self.reload_prompts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(200, self._process_gui_queue)

    def _build_ui(self):
        left = ttk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        dir_frame = ttk.LabelFrame(left, text="路徑設定 (Path Settings)")
        dir_frame.pack(anchor="w", fill="x", pady=5)
        pf = ttk.Frame(dir_frame)
        pf.pack(anchor="w", fill="x", padx=5, pady=5)
        ttk.Label(pf, text="Prompts 資料夾:").pack(side=tk.LEFT)
        ttk.Entry(pf, textvariable=self.prompts_dir_var).pack(
            side=tk.LEFT, expand=True, fill="x", padx=(4, 0)
        )
        ttk.Button(pf, text="瀏覽...", command=self._browse_prompts_dir).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        of = ttk.Frame(dir_frame)
        of.pack(anchor="w", fill="x", padx=5, pady=5)
        ttk.Label(of, text="輸出資料夾:").pack(side=tk.LEFT)
        ttk.Entry(of, textvariable=self.outputs_dir_var).pack(
            side=tk.LEFT, expand=True, fill="x", padx=(18, 0)
        )
        ttk.Button(of, text="瀏覽...", command=self._browse_outputs_dir).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        param_frame = ttk.LabelFrame(left, text="參數設定 (Parameters)")
        param_frame.pack(anchor="w", fill="x", pady=5)
        ttk.Label(param_frame, text="API 端點:").pack(anchor="w", padx=5)
        ttk.Entry(param_frame, textvariable=self.endpoint_var, width=48).pack(
            anchor="w", pady=2, padx=5, fill="x"
        )

        steps_label = ttk.Label(param_frame, text="步數 (Steps):")
        steps_label.pack(anchor="w", padx=5)
        steps_entry = ttk.Entry(param_frame, textvariable=self.steps_var, width=12)
        steps_entry.pack(anchor="w", pady=2, padx=5)
        ToolTip(steps_label, "生成圖片的迭代步數。\n建議值: 20-50")
        ToolTip(steps_entry, "生成圖片的迭代步數。\n建議值: 20-50")

        ttk.Label(param_frame, text="寬度 x 高度:").pack(anchor="w", padx=5)
        whf = ttk.Frame(param_frame)
        whf.pack(anchor="w", pady=2, padx=5)
        ttk.Entry(whf, textvariable=self.width_var, width=8).pack(side=tk.LEFT)
        ttk.Label(whf, text=" x ").pack(side=tk.LEFT)
        ttk.Entry(whf, textvariable=self.height_var, width=8).pack(side=tk.LEFT)

        cfg_label = ttk.Label(param_frame, text="CFG 比例 (Scale):")
        cfg_label.pack(anchor="w", padx=5)
        cfg_entry = ttk.Entry(param_frame, textvariable=self.cfg_var, width=12)
        cfg_entry.pack(anchor="w", pady=2, padx=5)
        ToolTip(cfg_label, "數值越高，圖片越貼近提示詞，但可能過於銳利。\n建議值: 7-12")
        ToolTip(cfg_entry, "數值越高，圖片越貼近提示詞，但可能過於銳利。\n建議值: 7-12")

        sampler_label = ttk.Label(param_frame, text="取樣器 (Sampler):")
        sampler_label.pack(anchor="w", padx=5)
        sampler_entry = ttk.Entry(param_frame, textvariable=self.sampler_var, width=20)
        sampler_entry.pack(anchor="w", pady=2, padx=5)
        ToolTip(sampler_label, "使用的取樣演算法，例如 DPM++ 2M Karras, Euler a 等。")
        ToolTip(sampler_entry, "使用的取樣演算法，例如 DPM++ 2M Karras, Euler a 等。")

        ttk.Label(param_frame, text="請求間延遲 (毫秒):").pack(anchor="w", padx=5)
        ttk.Entry(param_frame, textvariable=self.delay_var, width=12).pack(
            anchor="w", pady=2, padx=5
        )

        start_idx_label = ttk.Label(param_frame, text="手動設定起始點 (1 = 第 1 筆):")
        start_idx_label.pack(anchor="w", pady=(10, 0), padx=5)
        start_idx_entry = ttk.Entry(
            param_frame, textvariable=self.manual_start_index_var, width=12
        )
        start_idx_entry.pack(anchor="w", pady=2, padx=5)
        ToolTip(
            start_idx_label,
            "從指定的行號開始執行。\n會覆蓋從 progress.csv 自動恢復的進度。",
        )
        ToolTip(
            start_idx_entry,
            "從指定的行號開始執行。\n會覆蓋從 progress.csv 自動恢復的進度。",
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

        status_area = ttk.Frame(left)
        status_area.pack(anchor="w", fill="x", pady=(10, 0))
        ttk.Label(status_area, text="狀態:").pack(anchor="w")
        self.status_var = tk.StringVar(value="空閒 (Idle)")
        ttk.Label(status_area, textvariable=self.status_var, foreground="blue").pack(
            anchor="w"
        )
        ttk.Label(status_area, text="當前提示詞:").pack(anchor="w", pady=(8, 0))
        self._prompt_box = tk.Text(status_area, wrap=tk.WORD, height=4, width=40)
        self._prompt_box.pack(anchor="w", fill="x")
        self._prompt_box.config(state=tk.DISABLED)
        ttk.Label(status_area, text="日誌 (Log):").pack(anchor="w", pady=(8, 0))
        self.log_box = tk.Text(status_area, wrap=tk.WORD, height=6, width=40)
        self.log_box.pack(anchor="w", fill="x")
        self.log_box.config(state=tk.DISABLED)

        right = ttk.Frame(self.root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        ttk.Label(right, text="預覽 (Preview)").pack(anchor="w")
        self.preview_label = ttk.Label(right)
        self.preview_label.pack(anchor="center", pady=6)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progressbar = ttk.Progressbar(
            right, variable=self.progress_var, maximum=1.0
        )
        self.progressbar.pack(fill=tk.X, padx=6, pady=6)
        self.progress_text_var = tk.StringVar(value="0/0")
        ttk.Label(right, textvariable=self.progress_text_var).pack(anchor="center")

    def _browse_prompts_dir(self):
        path = filedialog.askdirectory(title="選擇 Prompts 資料夾")
        if path:
            self.prompts_dir_var.set(path)
            self.reload_prompts()

    def _browse_outputs_dir(self):
        path = filedialog.askdirectory(title="選擇輸出資料夾")
        if path:
            self.outputs_dir_var.set(path)
            self.reload_prompts()

    def _set_prompt_texts(self):
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

    # **優化**：限制日誌行數，避免記憶體問題
    def _log(self, s):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} {s}\n")

        num_lines = int(self.log_box.index("end-1c").split(".")[0])
        if num_lines > 200:  # 只保留最新的 200 行
            self.log_box.delete("1.0", f"{num_lines - 200}.0")

        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def reload_prompts(self):
        prompts_dir = self.prompts_dir_var.get()
        outputs_dir = self.outputs_dir_var.get()
        prompts_file = os.path.join(prompts_dir, "prompts.txt")
        neg_prompts_file = os.path.join(prompts_dir, "neg_prompts.txt")
        progress_csv_file = os.path.join(outputs_dir, "progress.csv")

        self.prompts = safe_prompt_list(prompts_file)
        self.neg_prompts = safe_prompt_list(neg_prompts_file)
        self.start_index = get_resume_index(progress_csv_file)

        self.manual_start_index_var.set(str(self.start_index + 1))
        self._set_prompt_texts()
        self._log(f"已從 '{prompts_dir}' 載入提示詞: {len(self.prompts)} 行。")
        self._log(
            f"從 '{progress_csv_file}' 恢復索引: {self.start_index} (第 {self.start_index + 1} 筆)"
        )

    def open_outputs(self):
        folder = os.path.abspath(self.outputs_dir_var.get())
        ensure_dir(folder)
        if sys.platform == "win32":
            os.startfile(folder)
        else:
            messagebox.showinfo("打開資料夾", f"輸出資料夾: {folder}")

    def open_progress_csv(self):
        path = os.path.abspath(os.path.join(self.outputs_dir_var.get(), "progress.csv"))
        if os.path.exists(path):
            if sys.platform == "win32":
                os.startfile(path)
            else:
                messagebox.showinfo("進度 CSV", f"進度 CSV: {path}")
        else:
            messagebox.showinfo(
                "進度 CSV", f"在輸出資料夾中找不到 progress.csv\n路徑: {path}"
            )

    # **優化**：重構 `on_start` 邏輯，使其更清晰
    def on_start(self):
        # 參數驗證
        try:
            if self.steps_var.get() <= 0:
                messagebox.showerror("參數錯誤", "步數 (Steps) 必須大於 0。")
                return
            if self.width_var.get() <= 0 or self.height_var.get() <= 0:
                messagebox.showerror("參數錯誤", "寬度和高度必須大於 0。")
                return
        except tk.TclError:
            messagebox.showerror(
                "參數錯誤", "請確保步數、寬高、延遲等參數為有效的數字。"
            )
            return

        prompts_dir = self.prompts_dir_var.get()
        outputs_dir = self.outputs_dir_var.get()
        ensure_dir(prompts_dir)
        ensure_dir(outputs_dir)

        # 重新載入提示詞，確保資料最新，並取得預設的恢復索引
        self.reload_prompts()

        # 決定最終的起始索引
        final_start_index = 0
        try:
            manual_input = int(self.manual_start_index_var.get())
            if manual_input >= 1:
                final_start_index = manual_input - 1
                self._log(
                    f"使用者手動設定起始點為第 {manual_input} 筆 (索引 {final_start_index})。"
                )
            else:
                final_start_index = 0
                self._log("手動輸入小於 1，將從頭開始 (索引 0)。")
        except ValueError:
            # 手動輸入無效，使用從 progress.csv 恢復的值
            final_start_index = self.start_index
            self._log(f"手動輸入無效，從 progress.csv 恢復索引: {final_start_index}。")

        if final_start_index >= len(self.prompts):
            messagebox.showwarning(
                "超出範圍",
                f"起始點 {final_start_index + 1} 超出提示詞總數 {len(self.prompts)}",
            )
            return

        self.start_index = final_start_index
        self._set_prompt_texts()  # 根據最終確定的索引更新提示詞預覽

        config = {
            "prompts_dir": prompts_dir,
            "outputs_dir": outputs_dir,
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

        self.worker = BatchWorker(
            self.prompts, self.neg_prompts, self.start_index, self.gui_queue, config
        )
        self.worker.start()

        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.resume_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.skip_btn.config(state=tk.NORMAL)
        self._log("工作者已啟動。")
        self.status_var.set("運行中 (Running)")

    def on_pause(self):
        if self.worker:
            self.worker.pause()
            self.pause_btn.config(state=tk.DISABLED)
            self.resume_btn.config(state=tk.NORMAL)
            self._log("已暫停。")
            self.status_var.set("已暫停 (Paused)")

    def on_resume(self):
        if self.worker:
            self.worker.resume()
            self.pause_btn.config(state=tk.NORMAL)
            self.resume_btn.config(state=tk.DISABLED)
            self._log("已恢復。")
            self.status_var.set("運行中 (Running)")

    def on_stop(self):
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
        if self.worker:
            self.worker.skip()
            self._log("已跳過當前索引。")

    def _process_gui_queue(self):
        try:
            while not self.gui_queue.empty():
                item = self.gui_queue.get_nowait()
                key, val = item[0], item[1]

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
                        self.progress_var.set((idx + 1) / total)
                elif key == "image_saved":
                    outpath = val
                    self._log(f"已儲存: {os.path.basename(outpath)}")
                    try:
                        pil = Image.open(outpath)
                        pil.thumbnail((512, 512))
                        tkimg = ImageTk.PhotoImage(pil)
                        self.preview_label.config(image=tkimg)
                        self.preview_label.image = tkimg
                    except Exception as e:
                        self._log(f"預覽載入失敗: {e}")
                elif key == "log":
                    self._log(val)
        except Exception as e:
            self._log(f"佇列處理錯誤: {e}")
        finally:
            self.root.after(200, self._process_gui_queue)

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if messagebox.askyesno("退出", "工作者正在運行。是否停止並退出?"):
                self.worker.stop()
                time.sleep(0.5)
            else:
                return
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
    root = tk.Tk()
    try:  # 在 Windows 上設定 'zoomed' 狀態
        root.state("zoomed")
    except tk.TclError:  # 在不支援的系統上（如某些 Linux WM）優雅地失敗
        root.geometry("1280x720")  # 設定一個較大的預設尺寸
    app = BatchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
