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
import tensorflow as tf
import ddsp
import ddsp.training
import gin

# --- Loudness ---
import librosa

import os
import pickle
import traceback

# ---- SAFE PATCH: allow non-divisible upsampling ----
import ddsp.core as _core
import numpy as _np


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
                 in_channel_indices=(2,3),  # 0-based: BlackHole 3/4 を想定
                 out_channel_map=(0,1),
                 sr=16000,
                 frame_ms=64,
                 hop_ms=16,
                 block_ms=16,
                 use_mps=True,
                 f0_model='full',
                 wet=1.0,
                 out_gain=1.0):
        self.stats = _load_stats(ckpt_dir)  # これを __init__ の最初の方に置く
        self.model = load_ddsp_model(ckpt_dir)
        self.sr = sr
        self.frame_size = int(sr * frame_ms / 1000)
        self.hop_size = int(sr * hop_ms / 1000)
        self.block_size = int(sr * block_ms / 1000)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_idx = in_channel_indices
        self.out_map = out_channel_map
        self.wet = float(wet)
        self.out_gain = float(out_gain)
        self.hop_ms = hop_ms
        self.add_endpoint = True

        # torch device
        if use_mps and torch.backends.mps.is_available():
            self.torch_device = torch.device("mps")
        elif torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
        else:
            self.torch_device = torch.device("cpu")

        # Queues
        self.in_q = queue.Queue(maxsize=64)
        self.out_q = queue.Queue(maxsize=64)

        # IO devices
        self.in_device = in_device
        self.out_device = out_device

        # worker thread
        self.stop_flag = threading.Event()
        self.worker = threading.Thread(target=self._process_loop, daemon=True)

        # simple latency report
        self.last_process_time = 0.0

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

            try:
                # 4) 特徴抽出（ここで例外が出たら次ループへ）
                f0 = extract_f0_torchcrepe(
                    x, self.sr, self.hop_size,
                    device=self.torch_device, model='full'
                )  # (T,)
                ld = extract_loudness_db(
                    x, self.sr, frame_size=self.frame_size, hop_size=self.hop_size,
                    ref_db=0.0, range_db=120.0
                )  # これで最低 -120dB まで落ちる

                # 5) 長さを揃える＆最低フレーム数のチェック
                T = min(len(f0), len(ld))
                if T < 2:  # add_endpoint=True だと T-1 区間なので1だと無理
                    continue
                f0 = f0[:T]
                ld = ld[:T]

                # 無声は f0==0 が返る前提でゲート
                voiced = (f0 > 0.0)
                ld[~voiced] = -120.0   # 無声フレームはラウドネスをガッツリ下げる
                # 余裕があれば念のため f0 もゼロのまま維持（既に0のはず）

                 # 6) テンソル化（(1,T,1)）
                f0_hz_tf = tf.convert_to_tensor(f0[np.newaxis, :, np.newaxis])
                ld_tf    = tf.convert_to_tensor(ld[np.newaxis, :, np.newaxis])

                # ======== ここから重要：T基準でn_samps、add_endpointは完全にFalseに統一 ========
                T   = int(f0_hz_tf.shape[1])                         # 例: 9
                spf = int(round(self.sr * (self.hop_ms / 1000.0)))   # 例: 128
                n_samps  = int(T * spf)                               # ← +1しない

                # 全シンセに一括反映（add_endpointもFalseに固定）
                hit = self._set_all_n_samples(n_samps, add_endpoint=False)
                print(f"[ddsp-rt] T={T}, spf={spf}, n_samps={n_samps}")
                print(f"[ddsp-rt] set n_samples on: {hit}")

                # features は f0 / loudness だけでOK（n_samplesはモジュール側で固定済み）
                features = {'f0_hz': f0_hz_tf, 'loudness_db': ld_tf}

                # 推論は1回だけ
                outputs = self.model(features, training=False)
                audio_tf = self._pick_audio(outputs)      # tf.Tensor [B, N] を期待
                y = tf.squeeze(audio_tf, axis=0).numpy()  # [N]

                # セーフティゲート（-85dBFS未満は完全ミュート）
                peak = np.max(np.abs(y)) + 1e-12
                if amp_to_db(peak) < -85.0:
                    y[:] = 0.0

                # ブロック長に整形してキューへ
                y_out = y[:self.block_size] if len(y) >= self.block_size else np.pad(y, (0, self.block_size - len(y)), mode='constant')
                self.out_q.put_nowait(y_out.astype(np.float32))

                if not getattr(self, "_shape_dbg", False):
                    print("[ddsp-rt] shapes", f0_hz_tf.shape, ld_tf.shape)
                    self._shape_dbg = True

            except Exception:
                print("[ddsp-rt] infer error:")
                traceback.print_exc()
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