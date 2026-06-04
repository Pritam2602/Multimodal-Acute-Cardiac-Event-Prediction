import os

targets = [
    r"D:\MINI_PROJECT\mimic_data\ecg_waveforms",
    r"D:\MINI_PROJECT\mimic_data\ecg_cache",
]

for d in targets:
    if os.path.isdir(d):
        total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(d) for f in fns)
        count = sum(len(fns) for _, _, fns in os.walk(d))
        print(f"{os.path.basename(d)}: {total/1e9:.1f} GB  ({count} files)")
    else:
        print(f"{os.path.basename(d)}: NOT FOUND")
