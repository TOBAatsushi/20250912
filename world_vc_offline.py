# world_vc_offline.py
import argparse, numpy as np, soundfile as sf
import pyworld as pw

def to_mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x

def warp_spectral(sp, ratio: float):
    if abs(ratio - 1.0) < 1e-6:
        return sp
    T, K = sp.shape
    idx = np.arange(K)
    src_idx = np.minimum((idx / ratio).astype(np.int32), K - 1)
    out = np.empty_like(sp)
    out[:] = sp[:, src_idx]
    return out

def tilt_brightness(sp, bright: float):
    # bright >0 で高域を持ち上げ、<0 で落とす。控えめカーブ。
    if abs(bright) < 1e-6:
        return sp
    K = sp.shape[1]
    w = np.linspace(-1.0, 1.0, K)  # -1:低域, +1:高域
    gain = np.exp(bright * w)      # だいたい ±0.3 程度まで推奨
    return sp * gain[None, :]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--pitch", type=float, default=0.0, help="半音数（+で上げる）")
    ap.add_argument("--formant", type=float, default=1.12, help="共鳴（1.0基準、>1で女声方向）")
    ap.add_argument("--bright", type=float, default=0.0, help="高域の明るさ（-0.3〜+0.3推奨）")
    ap.add_argument("--breath", type=float, default=0.15, help="息成分（0〜0.6くらい）")
    ap.add_argument("--f0_floor", type=float, default=140.0)
    ap.add_argument("--f0_ceil", type=float, default=1000.0)
    args = ap.parse_args()

    x, sr = sf.read(args.inp, dtype="float32")
    x = to_mono(x).astype(np.float64)

    # 解析
    # frame_period は 5.0ms に固定（十分に自然＆破綻しにくい）
    f0, t = pw.harvest(x, sr, f0_floor=args.f0_floor, f0_ceil=args.f0_ceil, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)   # スペクトル包絡
    apw = pw.d4c(x, f0, t, sr)         # 非周期性

    # ピッチ（半音）・フォルマント・明るさ・息
    if abs(args.pitch) > 1e-7:
        f0 = f0 * (2.0 ** (args.pitch / 12.0))
    if args.formant and abs(args.formant - 1.0) > 1e-6:
        sp = warp_spectral(sp, args.formant)
    if abs(args.bright) > 1e-6:
        sp = tilt_brightness(sp, args.bright)
    if args.breath > 0:
        apw = np.clip(apw + args.breath * (1.0 - apw), 0.0, 1.0)

    # 合成
    y = pw.synthesize(f0, sp, apw, sr, frame_period=5.0).astype(np.float32)

    # クリップ安全化
    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > 0.99:
        y = y / peak * 0.99

    sf.write(args.out, y, sr)
    print(f"done -> {args.out}  (sr={sr}, frames={len(y)})")

if __name__ == "__main__":
    main()