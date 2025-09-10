import argparse
import queue
import threading
import time
import numpy as np
import sounddevice as sd

# --- Torch/torchcrepe (F0) ---
import torch
import torchcrepe

# --- TensorFlow / DDSP ---
import ddsp
import ddsp.training
import gin

# --- Loudness ---
import librosa
import pickle
import traceback

# ---- SAFE PATCH: allow non-divisible upsampling ----
import ddsp.core as _core
import numpy as _np

import pyworld as pw

# --- TinyHarmNet ckpt 直読み 推論ヘルパー ---
import os, tensorflow as tf, ddsp, numpy as np


def _movavg_time(M, w=3):
    """時間軸移動平均（numpyのみでOK）"""
    if w <= 1: 
        return M
    T = M.shape[0]
    out = np.empty_like(M)
    half = w // 2
    # 累積和でO(T)に
    cs = np.cumsum(M, axis=0)
    for t in range(T):
        a = max(0, t - half)
        b = min(T - 1, t + half)
        s = cs[b] - (cs[a - 1] if a > 0 else 0)
        out[t] = s / float(b - a + 1)
    return out

def median_filter_1d(x, k=5):
    k = max(1, int(k) | 1)  # 奇数に
    r = k // 2
    pad = np.pad(x, (r, r), mode='edge')
    return np.array([np.median(pad[i:i+k]) for i in range(len(x))], dtype=np.float32)

def load_tiny_harmnet_infer_from_ckpt(ckpt_dir, sr=16000, hop=128):
    # 学習ログの統計値（手元のデータに合わせる）
    F0_MEAN, F0_STD = 58.37, 5.08
    LD_MEAN, LD_STD = -30.83, 9.97

    class _Pre(tf.keras.layers.Layer):
        def call(self, inputs, training=False):
            f0_hz = tf.convert_to_tensor(inputs["f0_hz"], tf.float32)
            ld_db = tf.convert_to_tensor(inputs["loudness_db"], tf.float32)
            f0_hz = tf.clip_by_value(f0_hz, 1.0, 8000.0)
            f0_m  = ddsp.core.hz_to_midi(f0_hz)
            return {
                "f0_hz": f0_hz,
                "f0_scaled": (f0_m - F0_MEAN) / (F0_STD + 1e-8),
                "ld_scaled": (ld_db - LD_MEAN) / (LD_STD + 1e-8),
            }

    class _Net(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.pre = _Pre()
            self.d1  = tf.keras.layers.Dense(128, activation="gelu")
            self.d2  = tf.keras.layers.Dense(128, activation="gelu")
            self.h_amp = tf.keras.layers.Dense(1)
            self.h_hd  = tf.keras.layers.Dense(32)  # 倍音数
        def call(self, inputs, training=False):
            f = self.pre(inputs, training=training)
            x = tf.concat([f["f0_scaled"], f["ld_scaled"]], axis=-1)
            h = self.d2(self.d1(x))
            amps = tf.nn.softplus(self.h_amp(h))        # [B,T,1]
            hd   = tf.nn.softmax(self.h_hd(h), axis=-1) # [B,T,32]
            return {"amps": amps, "hd": hd, "f0_hz": f["f0_hz"]}

    net = _Net()
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if not latest:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")
    tf.train.Checkpoint(model=net).restore(latest).expect_partial()

    # 1ブロック推論（numpy 1D f0/ld を受け取り、その長さTに合わせて合成）
    def infer(f0_np, ld_np):
        f0_np = np.asarray(f0_np, np.float32)
        ld_np = np.asarray(ld_np, np.float32)
        T = int(min(len(f0_np), len(ld_np)))
        if T < 2:
            return tf.zeros([1, 0], tf.float32)
        f0_tf = tf.constant(f0_np[:T][np.newaxis, :, np.newaxis], tf.float32)
        ld_tf = tf.constant(ld_np[:T][np.newaxis, :, np.newaxis], tf.float32)

        controls = net({"f0_hz": f0_tf, "loudness_db": ld_tf}, training=False)
        amps, hd, f0_hz = controls["amps"], controls["hd"], controls["f0_hz"]

        n_samps = int(T * hop)  # ← その時々のTで合成
        harm = ddsp.synths.Harmonic(sample_rate=sr, n_samples=n_samps)
        harm.add_endpoint = False
        audio = harm.get_signal(amplitudes=amps, harmonic_distribution=hd, f0_hz=f0_hz)  # [1, n_samps]
        return audio

    return infer


# -----------------------------------------------

def _force_n_samples(model, n):
    import ddsp.synths as _s
    hit = []
    seen = set()
    def _walk(x):
        if id(x) in seen:
            return
        seen.add(id(x))
        # Keras sublayers
        if hasattr(x, "layers"):
            for l in x.layers:
                _walk(l)
        # DDSP DAG modules
        if hasattr(x, "modules"):
            for m in x.modules:
                _walk(m)
        # 対象シンセへ n_samples を反映
        if isinstance(x, (_s.Harmonic, _s.FilteredNoise)):
            try:
                x.n_samples = int(n)
                hit.append(getattr(x, "name", x.__class__.__name__))
            except Exception:
                pass
    _walk(model)
    return hit

DEFAULT_STATS = {'f0_mean': 60.0, 'f0_std': 12.0, 'ld_mean': -45.0, 'ld_std': 10.0}

# --- フレーム数 T の取得を頑丈に（どれかの特徴量から推定） ---

def _infer_num_frames(features: dict):
    """features から [B, T, ...] の T を推定。なければ None"""
    candidate_keys = ("f0_hz", "loudness_db", "f0_scaled", "ld_scaled")
    for k in candidate_keys:
        x = features.get(k)
        if x is None:
            continue
        s = getattr(x, "shape", None)
        if s is None:
            continue
        # tf.TensorShape でも tuple でもOK
        try:
            rank = len(s)
        except TypeError:
            rank = None
        if rank and rank >= 2:
            return int(s[-2])  # [B, T, ...] の T
    return None

# dataset_statistics.pkl を読み込んで self.statsを入れているはずなので、ユーティリティを1個定義
def _hz_to_midi_safe(f0_hz: np.ndarray) -> np.ndarray:
    """0Hzを安全に扱いながらHz→MIDIに変換"""
    f = np.maximum(f0_hz, 1e-6)
    midi = 69.0 + 12.0 * np.log2(f / 440.0)
    midi = np.where(f0_hz > 0.0, midi, 0.0)
    return midi.astype(np.float32)

# 計読み込みユーティリティを追加
def _load_stats(ckpt_dir: str) -> dict:
    """dataset_statistics.pkl から f0/ld の平均・分散を取り出す。無ければデフォルト。"""
    stats = {'f0_mean': 60.0, 'f0_std': 12.0, 'ld_mean': -45.0, 'ld_std': 10.0}
    for name in ('dataset_statistics.pkl', 'dataset_statistics.pkl.backup'):
        p = os.path.join(ckpt_dir, name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, 'rb') as f:
                raw = pickle.load(f)
            if hasattr(raw, '__dict__'):
                raw = dict(raw.__dict__)

            def pick(d, keys, default):
                for k in keys:
                    if k in d and d[k] is not None:
                        return float(d[k])
                return default

            stats['f0_mean'] = pick(raw, ['f0_mean', 'f0_median', 'mean_f0_hz_midi'], stats['f0_mean'])
            stats['f0_std']  = pick(raw, ['f0_std', 'f0_scale', 'std_f0_hz_midi'],  stats['f0_std'])
            stats['ld_mean'] = pick(raw, ['ld_mean', 'loudness_db_mean', 'loudness_mean'], stats['ld_mean'])
            stats['ld_std']  = pick(raw, ['ld_std', 'loudness_db_std', 'loudness_std'],  stats['ld_std'])

            print(f"[ddsp-rt] stats loaded: f0({stats['f0_mean']:.1f},{stats['f0_std']:.1f}) "
                  f"ld({stats['ld_mean']:.1f},{stats['ld_std']:.1f})")
        except Exception as e:
            print(f"[ddsp-rt] WARN: failed to read stats: {e}")
        break
    return stats

# -----------------------------
# Utility
# -----------------------------
def db_to_amp(db):
    return np.power(10.0, db / 20.0)

def amp_to_db(amp):
    return 20.0 * np.log10(np.maximum(1e-5, amp))

def to_mono(x):
    if x.ndim == 2:
        return np.mean(x, axis=1)
    return x

def frame_audio(x, frame_size, hop_size):
    n = len(x)
    if n < frame_size:
        x = np.pad(x, (0, frame_size - n))
        n = frame_size
    frames = []
    for start in range(0, n - frame_size + 1, hop_size):
        frames.append(x[start:start+frame_size])
    return np.stack(frames, axis=0) if frames else np.zeros((0, frame_size), dtype=x.dtype)

# -----------------------------
# F0 & Loudness extractors
# -----------------------------
def extract_f0_torchcrepe(x, sr, hop_size, device, f0_min=50.0, f0_max=1100.0, model='full'):
    """
    x: 1D numpy float32 [-1,1]
    returns: f0_hz (T,)
    """
    # torchcrepe expects 16k or 44.1k; 16kに合わせておく前提
    # Hop は samples 単位 -> torchcrepe は "hop_length" サンプル
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N)
        f0 = torchcrepe.predict(
            t, sr, hop_length=hop_size,
            fmin=f0_min, fmax=f0_max,
            model=model, batch_size=1, device=device, decoder=torchcrepe.decode.viterbi
        ).squeeze(0)  # (T,)
        f0 = f0.detach().cpu().numpy().astype(np.float32)
    # Unvoiced frames become 0.0; DDSP 側で扱いやすいように 0 のままでOK
    return f0

def extract_loudness_db(x, sr, frame_size, hop_size, ref_db=20.7, range_db=80.0):
    """
    librosa.power_to_db ベースでラウドネス（dB）系列を取得し、DDSP推奨の正規化範囲に近づける。
    """
    # STFTベースのloudness: librosa の A-weighting ではなく簡易RMS
    S = librosa.feature.rms(y=x, frame_length=frame_size, hop_length=hop_size, center=True)  # (1, T)
    amp = S[0]
    loudness_db = amp_to_db(amp + 1e-8)

    # DDSP colab 互換っぽいスケーリング
    loudness_db = np.clip(loudness_db, ref_db - range_db, ref_db)
    return loudness_db.astype(np.float32)  # (T,)

# -----------------------------
# DDSP model loader
# -----------------------------
def load_ddsp_model(ckpt_dir):
    import os, pickle, gin, tensorflow as tf, ddsp

    from ddsp import synths as _synths
    from ddsp import processors as _procs

    gin_file   = os.path.join(ckpt_dir, "operative_config.gin")
    stats_path = os.path.join(ckpt_dir, "dataset_statistics.pkl")

    gin.clear_config()
    try:
        gin.enter_interactive_mode()
    except Exception:
        pass

    # ---------- ここを追加：旧ckpt互換のエイリアス登録（parse前！） ----------
    try:
        gin.external_configurable(_synths.Harmonic, name='synths.Additive')
        gin.external_configurable(_synths.Harmonic, name='Additive')
    except Exception:
        pass
    try:
        gin.external_configurable(_procs.Add, name='Add')  # KerasのAddと衝突回避
    except Exception:
        pass

    # （既に入れてある cumsum 互換や他の登録があれば、この直後に並べてOK）
    # 例：
    # def _cumsum_compat(x, axis=-1, exclusive=False, reverse=False, use_tpu=False):
    #     return tf.math.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse)
    # try: gin.external_configurable(_cumsum_compat, name='cumsum')
    # except: pass
    # -------------------------------------------------------------------------

    # 余計な .gin は無視しつつ最低限だけバインド
    extra_bindings = ['train.data_provider = None']
    gin.parse_config_files_and_bindings([gin_file], bindings=extra_bindings, skip_unknown=True)

    # ---- ランタイム前処理：f0_scaled/ld_scaled を作る（前回の案のまま） ----
    class _RuntimePreprocessor(tf.keras.layers.Layer):
        def __init__(self, stats_path=None, **kwargs):
            super().__init__(**kwargs)
            self.f0_mean, self.f0_std = 60.0, 12.0
            self.ld_mean, self.ld_std = -45.0, 10.0
            if stats_path and os.path.exists(stats_path):
                try:
                    with open(stats_path, "rb") as f:
                        stats = pickle.load(f)
                    def _get(k, dm, ds):
                        v = stats.get(k, None)
                        if hasattr(v, "mean") and hasattr(v, "std"):
                            return float(v.mean), float(v.std)
                        if isinstance(v, dict) and "mean" in v and "std" in v:
                            return float(v["mean"]), float(v["std"])
                        return dm, ds
                    self.f0_mean, self.f0_std = _get("f0_midi", self.f0_mean, self.f0_std)
                    self.ld_mean, self.ld_std = _get("loudness_db", self.ld_mean, self.ld_std)
                    print(f"[ddsp-rt] stats loaded: f0({self.f0_mean:.1f},{self.f0_std:.1f}) "
                          f"ld({self.ld_mean:.1f},{self.ld_std:.1f})")
                except Exception as e:
                    print("[ddsp-rt] stats load failed, using defaults:", e)
                    

        def call(self, features, training=False):
            f0_hz = tf.convert_to_tensor(features["f0_hz"], tf.float32)
            ld_db = tf.convert_to_tensor(features["loudness_db"], tf.float32)
            f0_hz = tf.clip_by_value(f0_hz, 1.0, 8000.0)
            f0_midi   = ddsp.core.hz_to_midi(f0_hz)
            f0_scaled = (f0_midi - self.f0_mean) / (self.f0_std + 1e-8)
            ld_scaled = (ld_db   - self.ld_mean) / (self.ld_std + 1e-8)
            return {"f0_scaled": f0_scaled, "ld_scaled": ld_scaled}
    # -------------------------------------------------------------------------

    # モデルを作って復元
    model = ddsp.training.models.Autoencoder()
    ckpt  = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    ckpt.restore(latest).expect_partial()

    # GinではなくPythonで前処理を差し替え（Gin構文エラー回避）
    model.preprocessor = _RuntimePreprocessor(stats_path=stats_path)

    return model

# ==== DDSP runtime helpers ====
def _walk_layers(obj):
    """Keras/Layers の木をyieldするジェネレータ"""
    if hasattr(obj, "layers"):
        for l in obj.layers:
            yield from _walk_layers(l)
    yield obj

def _set_model_n_samples(model, n_samps):
    """Additive/FilteredNoise など n_samples を持つプロセッサ全部に適用"""
    touched = []
    for layer in _walk_layers(model):
        # ddsp.synths.Harmonic は 'additive' という名前でいることが多い
        if hasattr(layer, "n_samples"):
            try:
                layer.n_samples = int(n_samps)
                touched.append(getattr(layer, "name", layer.__class__.__name__))
            except Exception:
                pass
    return touched

def _infer_T_from_features(features):
    """features から frame数 T を推定 (f0優先)"""
    f0 = features.get('f0_hz')
    if f0 is not None:
        return int(f0.shape[1])
    ld = features.get('loudness_db')
    if ld is not None:
        return int(ld.shape[1])
    return None
# =================================

# -----------------------------
# Real-time Engine
# -----------------------------
class DDSPRealtime:
    def __init__(self,
                 ckpt_dir,
                 in_device=None,
                 out_device=None,
                 in_channels=2,
                 out_channels=2,
                 in_channel_indices=(2,3),
                 out_channel_map=(0,1),
                 sr=16000,
                 frame_ms=32,
                 hop_ms=8,
                 block_ms=16,
                 use_mps=True,
                 f0_model='full',
                 wet=1.0,
                 out_gain=1.0,
                 silence_db=-70.0,
                 ld_boost_db=24.0,
                 mode='model',
                 pitch=0.0,
                 formant_ratio=1.0,
                 bright=0.0,
                 breath=0.0):
        self.stats = _load_stats(ckpt_dir)

        # ---- 先にサンプルレート関連を設定 ----
        self.sr = sr
        self.frame_size = int(sr * frame_ms / 1000)
        self.hop_size   = int(sr * hop_ms   / 1000)
        self.block_size = int(sr * block_ms / 1000)

        # ---- IO / パラメータ ----
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_idx = in_channel_indices
        self.out_map = out_channel_map
        self.wet = float(wet)
        self.out_gain = float(out_gain)
        self.hop_ms = hop_ms
        self.add_endpoint = True
        self.ld_boost_db = float(ld_boost_db)
        self.silence_db = float(silence_db)            # ← 引数そのまま尊重（バグ修正）
        self.mode = mode
        self.formant_ratio = float(formant_ratio)
        self.breath = float(breath)
        self.bright = float(bright)
        self._phi = 0.0                  # f0_sine用
        self.play_fifo = np.zeros(0, np.float32)
        self.silence_frames = max(3, int(round(120.0 / hop_ms)))
        self._silence_ctr = 0
        self.mode = str(mode)
        self.pitch = float(pitch)               # 半音指定（例：+4）
        self.formant_ratio = float(formant_ratio)
        self.bright = float(bright)
        self.breath = float(breath)
        self.frame_period_ms = float(hop_ms)   # WORLDのframe_periodをhop_msに統一
        self.ybuf = np.zeros(0, dtype=np.float32)  # 合成結果の貯蔵バッファ

        # ---- Torch デバイス ----
        if use_mps and torch.backends.mps.is_available():
            self.torch_device = torch.device("mps")
        elif torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
        else:
            self.torch_device = torch.device("cpu")

        # ---- Queues / Devices ----
        self.in_q = queue.Queue(maxsize=64)
        self.out_q = queue.Queue(maxsize=64)
        self.in_device = in_device
        self.out_device = out_device

        # ---- ここで ckpt 直読みの推論器を作る ----
        self.ckpt_dir = ckpt_dir
        self.infer = load_tiny_harmnet_infer_from_ckpt(
            self.ckpt_dir, sr=self.sr, hop=self.hop_size
        )
        # ---- worker thread 準備 ----
        self.stop_flag = threading.Event()
        self.worker = threading.Thread(target=self._process_loop, daemon=True)
        self.last_process_time = 0.0
        self.play_fifo = np.zeros(0, dtype=np.float32)


    def _set_all_n_samples(self, n_samps: int, add_endpoint: bool = False):
        """モデル配下の全プロセッサに n_samples / add_endpoint を伝搬して、触れた名前を返す"""
        seen, hit = set(), []

        def _set(obj):
            if obj is None or id(obj) in seen:
                return
            seen.add(id(obj))

            if hasattr(obj, "n_samples"):
                try:
                    obj.n_samples = int(n_samps)
                    hit.append(getattr(obj, "name", obj.__class__.__name__))
                except Exception:
                    pass
            if hasattr(obj, "add_endpoint"):
                try:
                    obj.add_endpoint = bool(add_endpoint)
                except Exception:
                    pass

            # ツリーを全部たどる
            pg = getattr(obj, "processor_group", None)
            if pg is not None:
                _set(pg)

            dag = getattr(obj, "dag", None)
            if dag is not None and hasattr(dag, "modules"):
                mods = getattr(dag, "modules", None)
                if isinstance(mods, dict):
                    for m in mods.values():
                        _set(m)

            procs = getattr(obj, "processors", None)
            if procs:
                for p in procs:
                    proc = getattr(p, "processor", p)
                    _set(proc)

            modules = getattr(obj, "modules", None)
            if isinstance(modules, dict):
                for m in modules.values():
                    _set(m)

            layers = getattr(obj, "layers", None)
            if isinstance(layers, (list, tuple)):
                for l in layers:
                    _set(l)

        _set(self.model)
        return hit

    def start(self):
        self.worker.start()
        self.stream = sd.Stream(
            samplerate=self.sr,
            blocksize=self.block_size,
            dtype='float32',
            channels=max(self.in_channels, self.out_channels),
            device=(self.in_device, self.out_device),
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        self.stop_flag.set()
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            # XRuns 等
            pass

        # 入力：指定 ch (例: 3/4) → モノラル合成
        # indata shape: (frames, channels)
        take = []
        for ch in self.in_idx:
            if ch < indata.shape[1]:
                take.append(indata[:, ch])
        if len(take) == 0:
            mono = np.zeros(frames, dtype=np.float32)
        else:
            mono = np.mean(np.stack(take, axis=1), axis=1).astype(np.float32)

        # enqueue
        try:
            self.in_q.put_nowait(mono.copy())
        except queue.Full:
            pass

        # 出力：処理済みがあれば、それを L/R に配置、なければスルー
        if not self.out_q.empty():
            y = self.out_q.get_nowait()
        else:
            y = np.zeros(frames, dtype=np.float32)

        # Wet/Dry は Max 側のクロスフェーダでやる想定だが、ここでも wet を適用可
        y = self.out_gain * (self.wet * y)

        # マッピング（out_map の ch に y を、その他は 0）
        outdata[:] = 0.0
        for i, ch in enumerate(self.out_map):
            if ch < outdata.shape[1]:
                outdata[:, ch] = y

    def _pick_audio(self, outputs):
        # 1) テンソルそのものが返ってきたらそれを使う
        if isinstance(outputs, tf.Tensor):
            return outputs  # shape: [B, N] 期待

        # 2) よくあるキー名を優先して拾う
        if isinstance(outputs, dict):
            preferred = [
                'audio', 'signal',                   # まずは標準っぽい
                'processor_group/signal', 'add/signal', 'sum/signal',
                'output_signal'
            ]
            for k in preferred:
                if k in outputs:
                    return outputs[k]

            # 3) /signal で終わるモジュール出力が複数あれば全部足す（Additive + Noise など）
            parts = [v for k, v in outputs.items()
                    if isinstance(k, str) and k.endswith('/signal')]
            if parts:
                return tf.add_n(parts)

            # 4) それでも無ければ、[B, N] 形状の浮動小数テンソルを探して最初の1個を使う
            for k, v in outputs.items():
                try:
                    if isinstance(v, tf.Tensor) and len(v.shape) == 2 and int(v.shape[0]) == 1 and v.dtype.is_floating:
                        return v
                except Exception:
                    pass

        # 5) 何も見つからないときはキー一覧を出して失敗
        raise KeyError(f"No audio-like tensor in outputs. keys={list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)}")

    def _process_loop(self):
        import traceback
        buf = np.zeros(0, dtype=np.float32)
        target_len = max(self.frame_size + self.hop_size * 2, self.block_size * 4)

        while not self.stop_flag.is_set():
            # 1) 入力ブロックの取得
            try:
                block = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            buf = np.concatenate([buf, block], axis=0)

            # 2) まとまった長さになるまで貯める
            if len(buf) < target_len:
                continue

            # 3) 切り出し＆前処理
            x = buf[:target_len]
            buf = buf[self.hop_size:]  # オーバーラップ前進
            x = np.clip(x, -1.0, 1.0).astype(np.float32)
            rms = float(np.sqrt(np.mean(x**2) + 1e-12))
            in_db = 20.0*np.log10(max(rms, 1e-6))
            self._silence_ctr = self._silence_ctr + 1 if in_db < self.silence_db else 0
            if self._silence_ctr >= self.silence_frames:
                self.play_fifo = np.zeros(0, dtype=np.float32)
                try: self.out_q.put_nowait(np.zeros(self.block_size, dtype=np.float32))
                except queue.Full: pass
                continue

            try:
                # 4) 特徴抽出
                f0 = extract_f0_torchcrepe(
                    x, self.sr, self.hop_size,
                    device=self.torch_device, model='full'
                )  # (T,)
                ld = extract_loudness_db(
                    x, self.sr,
                    frame_size=self.frame_size, hop_size=self.hop_size,
                    ref_db=0.0, range_db=120.0
                )

                # 5) 長さ揃え・最低フレーム数
                T = min(len(f0), len(ld))
                if T < 2:
                    # 足りないときは無音を1ブロックだけ出して継続
                    try:
                        self.out_q.put_nowait(np.zeros(self.block_size, dtype=np.float32))
                    except queue.Full:
                        pass
                    continue
                f0 = f0[:T]
                ld = ld[:T]

                # ====== モード分岐 ======
                if self.mode == 'passthru':
                    y_out = block[:self.block_size].astype(np.float32)
                    self.out_q.put_nowait(y_out)
                    continue

                elif self.mode == 'f0_sine':
                    voiced = (f0 > 0.0)
                    f0 = np.clip(f0, 70.0, 500.0)
                    if len(f0) >= 5:
                        f0[1:] = 0.8*f0[1:] + 0.2*f0[:-1]
                    amp = 10.0 ** (np.clip(ld, -24.0, 0.0) / 20.0)
                    a_s = np.repeat(amp, self.hop_size)[:self.block_size]
                    a_s = np.maximum(a_s, 10 ** (-12/20))
                    f_s = np.repeat(f0, self.hop_size)[:self.block_size]
                    phase_inc = 2.0*np.pi*f_s/float(self.sr)
                    phi = (self._phi + np.cumsum(phase_inc)) % (2.0*np.pi)
                    self._phi = float(phi[-1])
                    y = np.sin(phi).astype(np.float32) * a_s.astype(np.float32)
                    self.out_q.put_nowait(y)
                    continue

                elif self.mode == 'world_vc':
                    # === WORLD 解析条件を完全に統一 ===
                    frame_period = self.frame_period_ms  # ms。harvest と synthesize を一致させる

                    # 入力レベルが小さすぎる時は解析をスキップ（ゲートは“ゆっくり”）
                    rms = float(np.sqrt(np.mean(x.astype(np.float64)**2) + 1e-12))
                    rms_db = 20.0 * np.log10(max(rms, 1e-12))
                    if rms_db < float(self.silence_db):
                        # 何も合成せずに“無音”だけ出して次へ
                        silent = np.zeros(self.block_size, dtype=np.float32)
                        try:
                            self.out_q.put_nowait(silent)
                        except queue.Full:
                            pass
                        continue

                    x64 = x.astype(np.float64)

                    # ① f0推定（女声寄りレンジ）→ stonemaskで精緻化
                    f0_w, t_w = pw.harvest(
                        x64, self.sr,
                        f0_floor=140.0, f0_ceil=1000.0,
                        frame_period=frame_period
                    )
                    f0_w = pw.stonemask(x64, f0_w, t_w, self.sr)

                    # ② スペクトル・非周期性（※ cheaptrick/d4c は frame_period 引数を取らない版）
                    sp = pw.cheaptrick(x64, f0_w, t_w, self.sr)  # [T, K]
                    ap = pw.d4c(      x64, f0_w, t_w, self.sr)   # [T, K]

                    # ③ 軽めの時間平滑（ギラつき・ジャリ感を抑える）
                    f0_w = _movavg_time(f0_w[:, None], w=3)[:, 0]
                    sp   = _movavg_time(sp, w=3)
                    ap   = _movavg_time(ap, w=3)

                    # ④ ピッチ／フォルマント／明るさ／ブレスの適用（今のロジックそのまま使う想定）
                    #    もし既存の処理が前後にあるなら、それをここに移してから synthesize へ。

                    # ⑤ 合成（※ harvest と同じ frame_period を必ず渡す）
                    y_world = pw.synthesize(f0_w, sp, ap, self.sr, frame_period=frame_period).astype(np.float32)

                    # ⑥ 連続化：合成波を貯めて、block_size 単位で切り出して供給
                    self.ybuf = np.concatenate([self.ybuf, y_world], axis=0)
                    while len(self.ybuf) >= self.block_size:
                        chunk = self.ybuf[:self.block_size]
                        self.ybuf = self.ybuf[self.block_size:]
                        # 出力ゲインとwetは後段で適用される想定ならここは素のままでOK
                        try:
                            self.out_q.put_nowait(chunk)
                        except queue.Full:
                            # あふれたら捨てる（リアルタイム優先）
                            pass

                else:
                    # 既存の TinyHarmNet ルート（必要なら残す）
                    voiced = (f0 > 0.0)
                    ld2 = ld.copy()
                    ld2[~voiced] = -60.0
                    y_tf = self.infer(f0, ld2)
                    y = y_tf.numpy()[0]
                    if len(y) < self.block_size:
                        y = np.pad(y, (0, self.block_size-len(y)))
                    self.out_q.put_nowait(y[:self.block_size].astype(np.float32))
                    continue
                # ====== /モード分岐 ======

                # f0 の安全範囲と軽い平滑
                f0 = np.clip(f0, 70.0, 800.0)
                if len(f0) >= 5:
                    f0[1:] = 0.8*f0[1:] + 0.2*f0[:-1]  # 1ステップ遅れブレンド

                voiced = (f0 > 0.0)

                # ピッチは安全範囲にクリップ＆軽い平滑
                f0 = np.clip(f0, 70.0, 800.0)
                if len(f0) >= 5:
                    f0[1:] = 0.8 * f0[1:] + 0.2 * f0[:-1]

                # 無声は強めに落とす（が、あとで floor を入れる）
                ld[~voiced] = -90.0

                # ここから env（包絡）を作るが、床を作って掛け過ぎを防ぐ
                ld_for_env = np.pad(ld, (0, max(0, 250 - len(ld))), mode='edge')[:250]
                env = 10.0 ** (np.clip(ld_for_env, -30.0, 0.0) / 20.0)   # ← 最低 -30dB までに制限

                # 6) 前処理（無声ゲート＆軽い平滑）
                voiced = (f0 > 0.0)
                if np.mean(voiced) < 0.05:
                    # ほぼ無声なら無音を1ブロック出して継続
                    try:
                        self.out_q.put_nowait(np.zeros(self.block_size, dtype=np.float32))
                    except queue.Full:
                        pass
                    continue

                # f0 簡易平滑（1サンプル遅れとのブレンド）
                f0_sm = f0.copy()
                f0_sm[1:] = 0.8 * f0[1:] + 0.2 * f0[:-1]
                f0 = f0_sm

                # （オプション）ピッチ微調整：__init__ に self.pitch_shift を仕込んでいる場合だけ
                if hasattr(self, "pitch_shift") and self.pitch_shift != 0.0:
                    f0 = f0 * (2.0 ** (self.pitch_shift / 12.0))

                # ラウドネス移動平均（5点）
                if len(ld) >= 5:
                    ld = np.convolve(ld, np.ones(5, dtype=np.float32) / 5.0, mode='same')
                # 無声フレームは強めに落とす
                ld[~voiced] = -90.0

                # 7) 推論（ckpt直読みの TinyHarmNet）
                y_tf = self.infer(f0, ld)   # shape [1, N_SAMPLES]
                y = y_tf.numpy().reshape(-1).astype(np.float32)
                env_samp = np.repeat(env, self.hop_size)[:len(y)]
                env_samp = np.maximum(env_samp, 10 ** (-12 / 20))  # -12dB の床
                y *= env_samp.astype(np.float32)

                # 8) FIFO にためて、block_size ごとに吐く
                self.play_fifo = np.concatenate([self.play_fifo, y]).astype(np.float32)
                while len(self.play_fifo) >= self.block_size:
                    y_out = self.play_fifo[:self.block_size]
                    self.play_fifo = self.play_fifo[self.block_size:]
                    try:
                        self.out_q.put_nowait(y_out.copy())
                    except queue.Full:
                        # あふれたら最新優先で古いバッファを破棄
                        self.play_fifo = np.zeros(0, dtype=np.float32)
                        break

            except Exception:
                print("[ddsp-rt] infer error:")
                traceback.print_exc()
                # 失敗時も無音を1ブロック出して継続
                try:
                    self.out_q.put_nowait(np.zeros(self.block_size, dtype=np.float32))
                except Exception:
                    pass
                continue
                        
# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True, help='DDSP checkpoint directory')
    parser.add_argument('--gin', default=None, help='Path to operative_config.gin (optional)')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--frame_ms', type=float, default=64.0)
    parser.add_argument('--hop_ms', type=float, default=16.0)
    parser.add_argument('--block_ms', type=float, default=16.0)
    parser.add_argument('--in_dev', default=None, help='Sounddevice input device name or index (BlackHole)')
    parser.add_argument('--out_dev', default=None, help='Sounddevice output device name or index (BlackHole)')
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--out_channels', type=int, default=2)
    parser.add_argument('--in_ch', type=int, nargs='+', default=[2,3], help='Input channel indices (0-based), e.g., 2 3 for ch 3/4')
    parser.add_argument('--out_map', type=int, nargs='+', default=[0,1], help='Output channel map indices (0-based) for wet')
    parser.add_argument('--no_mps', action='store_true', help='Disable Apple MPS for torchcrepe')
    parser.add_argument('--wet', type=float, default=1.0)
    parser.add_argument('--gain', type=float, default=1.0)
    parser.add_argument('--pitch', type=float, default=0.0, help='Pitch shift in semitones')
    parser.add_argument('--silence_db', type=float, default=-70.0)
    parser.add_argument('--mode', choices=['model','f0_sine','passthru','world_vc'], default='world_vc')
    parser.add_argument('--formant_ratio', type=float, default=1.05, help='>1で女声寄り、<1で男声寄り（0.85–1.25推奨）')
    parser.add_argument('--breath', type=float, default=0.15, help='息成分（0.0–0.5）')
    parser.add_argument('--bright', type=float, default=0.0, help='明るさ(dB/Oct相当の傾き、-3.0〜+3.0）')
    args = parser.parse_args()

    # Device listing helper
    if args.in_dev is None or args.out_dev is None:
        print("=== Available devices ===")
        print(sd.query_devices())
        print("=========================")

    engine = DDSPRealtime(
        ckpt_dir=args.ckpt_dir,
        in_device=args.in_dev,
        out_device=args.out_dev,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        in_channel_indices=tuple(args.in_ch),
        out_channel_map=tuple(args.out_map),
        sr=args.sr,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        block_ms=args.block_ms,
        use_mps=(not args.no_mps),
        wet=args.wet,
        out_gain=args.gain,
        mode=args.mode,
        pitch=args.pitch,
        formant_ratio=args.formant_ratio,
        bright=args.bright,
        breath=args.breath,
        silence_db=args.silence_db,
    )

    try:
        engine.start()
        print("[ddsp-rt] running... Ctrl+C to stop")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        print("[ddsp-rt] stopped.")

if __name__ == '__main__':
    main()