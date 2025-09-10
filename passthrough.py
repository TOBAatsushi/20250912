import sounddevice as sd
import numpy as np
import torch, torchcrepe, threading, time
from collections import deque

# ====== Audio / Device ======
SR       = 48000
BLOCK    = 256
DEV      = 8              # BlackHole 16ch のデバイス番号
INMAP    = [3, 4]         # Max→BH: ch3/4
OUTMAP   = [1, 2]         # BH→Max: ch1/2

# ====== CREPE (低負荷寄り設定) ======
HOP      = 480            # 10ms @48k（計算負荷↓＆ジッター↓）
F0_MIN   = 65.0
F0_MAX   = 1000.0
MODEL    = "full"         # 'tiny'：速い / 'full'：精度↑
DEVICE   = "mps"          # M4なら 'mps' もOK。まずは 'cpu' で安定確認

# ====== Smoothing / Voicing ======
VOICING_THR = 0.08        # 旧より緩め
MED_N       = 7           # メディアン窓（フレーム数）
ALPHA       = 0.25        # ワンポールLPF係数（大きいほど追従↑）

# ====== Harmonic Synth ======
N_HARM   = 24
HARM_POW = 1.25
WET_GAIN = 2

# ====== Shared state ======
# オーディオ→推論用の入力バッファ（約 200ms=9600サンプル分）
inbuf = deque(maxlen= SR//5)
# 直近の f0（連続出音のため常に何か入れる）
f0_lock   = threading.Lock()
f0_smooth = 220.0

# L/R の位相
phase_lr = np.zeros((2, N_HARM), dtype=np.float64)
TWO_PI   = 2.0 * np.pi

# ====== f0 worker thread ======
def f0_worker():
    global f0_smooth
    hist = deque(maxlen=MED_N)
    while True:
        # 十分たまってから処理（200ms 窓の末尾から 10ms hop で計算）
        if len(inbuf) < SR//5:
            time.sleep(0.005)
            continue

        # コピーして安全に処理
        x = np.array(inbuf, dtype=np.float32)
        audio = torch.from_numpy(x).unsqueeze(0)  # (1, T)

        with torch.no_grad():
            f0, pd = torchcrepe.predict(
                audio=audio,
                sample_rate=SR,
                hop_length=HOP,         # 10ms
                fmin=F0_MIN, fmax=F0_MAX,
                model=MODEL,            # 旧API：文字列でOK
                batch_size=1,
                device=DEVICE,
                return_periodicity=True
            )
            f0 = f0[0].cpu().numpy()
            pd = pd[0].cpu().numpy()
            f0[pd < VOICING_THR] = 0.0

        # 末尾（最新）フレームを使用
        f_last = float(f0[-1])
        if f_last > 0.0:
            hist.append(f_last)
            f_med = float(np.median(list(hist)))
            # ワンポール平滑
            f_new = ALPHA * f_med + (1 - ALPHA) * f0_smooth
            f_new = float(np.clip(f_new, F0_MIN, F0_MAX))
            with f0_lock:
                f0_smooth = f_new
        # 無声でも f0_smooth は保持（出音を止めない）

        time.sleep(0.005)

# ====== Synth ======
def synth_block(f0_hz: float, amp: float, frames: int) -> np.ndarray:
    """ブロック一定 f0 前提、位相はブロック終端で更新。連続位相でブツ切れ防止。"""
    if amp <= 1e-7:
        return np.zeros((frames, 2), dtype=np.float32)

    # 動的アンチエイリアス
    nyq = SR * 0.5 - 200.0
    n_max = int(np.floor(nyq / max(1.0, f0_hz)))
    n_harm = int(np.clip(min(N_HARM, n_max), 1, N_HARM))

    n = np.arange(frames, dtype=np.float64)
    yL = np.zeros(frames, dtype=np.float64)
    yR = np.zeros(frames, dtype=np.float64)

    for h in range(1, n_harm + 1):
        fh = f0_hz * h
        omega = TWO_PI * fh / SR

        # ブロック開始位相を固定して合成（時間方向連続）
        phiL0 = phase_lr[0, h-1]
        phiR0 = phase_lr[1, h-1] + 0.3

        yL += (amp / (h**HARM_POW)) * np.sin(omega * n + phiL0)
        yR += (amp / (h**HARM_POW)) * np.sin(omega * n + phiR0)

        # 次ブロック用にまとめて位相更新
        phase_lr[0, h-1] = (phiL0 + omega * frames) % (2*np.pi)
        phase_lr[1, h-1] = (phiR0 + omega * frames) % (2*np.pi)

    scale = 1.0 / np.sqrt(max(1, n_harm))
    y = np.stack([yL*scale, yR*scale], axis=1).astype(np.float32)
    y *= WET_GAIN
    return y

# ====== Audio callback ======
last_log = 0.0
def cb(indata, outdata, frames, timeinfo, status):
    global last_log
    if status:
        print(status)

    # ステレオ→モノ（推論は別スレッド）
    mono = indata.mean(axis=1).astype(np.float32)
    inbuf.extend(mono)

    # 入力RMS（たまに表示）
    if time.time() - last_log > 0.5:
        rms = float(np.sqrt(np.mean(mono*mono) + 1e-12))
        print(f"RMS: {rms:.4f}  f0_smooth: {f0_smooth:.1f} Hz  (MODEL={MODEL}, DEVICE={DEVICE})")
        last_log = time.time()

    # ブロックの出力レベル（RMS→軽い圧縮）
    amp = float(np.tanh(3.0 * np.sqrt(np.mean(mono*mono) + 1e-12)))

    # 現在の滑らかな f0 を使って常に“鳴らす”
    with f0_lock:
        f0_cur = float(f0_smooth)

    outdata[:] = synth_block(f0_cur, amp, frames)

# ====== Start ======
if __name__ == "__main__":
    # f0 推論スレッド起動
    t = threading.Thread(target=f0_worker, daemon=True)
    t.start()

    in_settings  = sd.CoreAudioSettings(channel_map=INMAP)
    out_settings = sd.CoreAudioSettings(channel_map=OUTMAP)

    with sd.Stream(samplerate=SR, blocksize=BLOCK, dtype='float32',
                   channels=2, device=(DEV, DEV),
                   extra_settings=(in_settings, out_settings),
                   latency='low', callback=cb):
        print("Running threaded DDSP-ish… Ctrl+C で停止")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            pass