import pandas as pd, numpy as np, matplotlib.pyplot as plt, warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (16, 10)

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
print(f"Total trajectories: {len(traj):,}")
print(f"Distribution: {traj['trajectory'].value_counts().to_dict()}")

# ── 1. Threshold sensitivity (using traj first_VAF/last_VAF directly) ──
print("\n" + "=" * 60)
print("1. THRESHOLD SENSITIVITY")
print("=" * 60)

def reclassify(first, last, inc_r, dec_r):
    if last == 0 and first == 0:
        return "undetected"
    if last > inc_r * first:
        return "increasing"
    if last < dec_r * first:
        return "decreasing"
    return "stable"

thresholds_inc = [1.2, 1.5, 2.0, 3.0, 5.0]
thresholds_dec = [0.8, 0.5, 0.3, 0.2, 0.1]

total = len(traj)
for inc_t in thresholds_inc:
    for dec_t in thresholds_dec:
        inc_c = ((traj["last_VAF"] > inc_t * traj["first_VAF"]) & (traj["first_VAF"] > 0)).sum()
        dec_c = ((traj["last_VAF"] < dec_t * traj["first_VAF"]) & (traj["first_VAF"] > 0)).sum()
        sta_c = total - inc_c - dec_c - ((traj["first_VAF"] == 0) & (traj["last_VAF"] == 0)).sum()
        print(f"  inc>{inc_t:.1f}x dec<{dec_t:.1f}x -> "
              f"inc={inc_c:>5d}({inc_c/total*100:>4.1f}%) "
              f"dec={dec_c:>5d}({dec_c/total*100:>4.1f}%) "
              f"sta={sta_c:>5d}({sta_c/total*100:>4.1f}%)")

# ── 2. Edge cases ──
print("\n" + "=" * 60)
print("2. EDGE CASES")
print("=" * 60)

# Single TP variants (n_timepoints=1)
single = traj[traj["n_timepoints"] == 1]
print(f"  Single-TP trajectories: {len(single)} - classified as 'stable' but uninformative")

# 2TP borderline
two = traj[traj["n_timepoints"] == 2].copy()
two["ratio"] = two["last_VAF"] / two["first_VAF"].clip(0.001)
close_inc = two[(two["ratio"] > 1.3) & (two["ratio"] < 1.7)]
close_dec = two[(two["ratio"] > 0.3) & (two["ratio"] < 0.7)]
print(f"  Ambiguous inc (ratio 1.3-1.7, 2TP): {len(close_inc)}")
print(f"  Ambiguous dec (ratio 0.3-0.7, 2TP): {len(close_dec)}")

# First VAF very small
tiny = traj[(traj["trajectory"] == "increasing") & (traj["first_VAF"] < 0.005)]
print(f"  'Increasing' with first_VAF < 0.005: {len(tiny)}")

# Decreasing to exactly 0
dec_zero = ((traj["trajectory"] == "decreasing") & (traj["last_VAF"] == 0)).sum()
print(f"  Decreasing to exactly 0: {dec_zero:,}")

# ── 3. Noise injection ──
print("\n" + "=" * 60)
print("3. NOISE INJECTION TEST")
print("=" * 60)

np.random.seed(42)
stable_pool = traj[(traj["trajectory"] == "stable") & (traj["n_timepoints"] >= 3)].copy()
n_sample = min(500, len(stable_pool))
stable_sample = stable_pool.sample(n_sample, random_state=42).reset_index(drop=True)

for noise_level in [0.01, 0.025, 0.05, 0.10]:
    noisy_last = stable_sample["last_VAF"].values + np.random.normal(0, noise_level, n_sample)
    noisy_last = np.clip(noisy_last, 0, 1)
    inc_flip = (noisy_last > 1.5 * stable_sample["first_VAF"].values).sum()
    dec_flip = (noisy_last < 0.5 * stable_sample["first_VAF"].values).sum()
    flip_rate = (inc_flip + dec_flip) / n_sample * 100
    print(f"  Noise sd={noise_level:.3f}: {flip_rate:.1f}% flip - inc={inc_flip} dec={dec_flip}")

# ── 4. Threshold sweep for plots ──
print("\n" + "=" * 60)
print("4. GENERATING FIGURE")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel A: inc threshold sweep
inc_rates, dec_rates, sta_rates = [], [], []
for inc_t in [1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0, 5.0]:
    ic = ((traj["last_VAF"] > inc_t * traj["first_VAF"]) & (traj["first_VAF"] > 0)).sum()
    dc = ((traj["last_VAF"] < 0.5 * traj["first_VAF"]) & (traj["first_VAF"] > 0)).sum()
    sc = total - ic - dc
    inc_rates.append(ic/total*100)
    dec_rates.append(dc/total*100)
    sta_rates.append(sc/total*100)

axes[0,0].plot([1.1,1.2,1.3,1.4,1.5,1.7,2.0,2.5,3.0,5.0], inc_rates, "ro-", label="increasing")
axes[0,0].plot([1.1,1.2,1.3,1.4,1.5,1.7,2.0,2.5,3.0,5.0], dec_rates, "bs-", label="decreasing")
axes[0,0].axvline(1.5, color="r", ls="--", alpha=0.4)
axes[0,0].set_xlabel("inc threshold (x first VAF)"); axes[0,0].set_ylabel("% of trajectories")
axes[0,0].set_title("A) Increasing threshold sweep"); axes[0,0].legend()

# Panel B: dec threshold sweep
inc_r2, dec_r2, sta_r2 = [], [], []
for dec_t in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    ic = ((traj["last_VAF"] > 1.5 * traj["first_VAF"]) & (traj["first_VAF"] > 0)).sum()
    dc = ((traj["last_VAF"] < dec_t * traj["first_VAF"]) & (traj["first_VAF"] > 0)).sum()
    sc = total - ic - dc
    inc_r2.append(ic/total*100)
    dec_r2.append(dc/total*100)
    sta_r2.append(sc/total*100)

axes[0,1].plot([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1], dec_r2, "bs-", label="decreasing")
axes[0,1].axvline(0.5, color="b", ls="--", alpha=0.4)
axes[0,1].invert_xaxis()
axes[0,1].set_xlabel("dec threshold (x first VAF)"); axes[0,1].set_ylabel("% of trajectories")
axes[0,1].set_title("B) Decreasing threshold sweep"); axes[0,1].legend()

# Panel C: Ratio distribution for 2-TP trajectories
two_tp = traj[traj["n_timepoints"] == 2].copy()
two_tp["log2_ratio"] = np.log2((two_tp["last_VAF"] + 1e-6) / (two_tp["first_VAF"] + 1e-6))
log2_vals = two_tp["log2_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
axes[0,2].hist(log2_vals, bins=60, color="steelblue", edgecolor="white")
axes[0,2].axvline(np.log2(1.5), color="red", ls="--", lw=2, label="inc threshold (1.5x)")
axes[0,2].axvline(np.log2(0.5), color="blue", ls="--", lw=2, label="dec threshold (0.5x)")
axes[0,2].set_xlabel("log2(last_VAF / first_VAF)"); axes[0,2].set_ylabel("Count")
axes[0,2].set_title("C) Ratio distribution (2-TP trajectories)"); axes[0,2].legend()

# Panel D: Borderline examples
border = close_inc.head(3)
for _, row in border.iterrows():
    vaf = [row["first_VAF"], row["last_VAF"]]
    axes[1,0].plot([0, 1], vaf, "o-", label=f"{row['variant_id'].split(' | ')[-1][:20]} ratio={row['ratio']:.2f}")
axes[1,0].set_title("D) Borderline increasing (ratio 1.3-1.7)"); axes[1,0].legend(fontsize=7)
axes[1,0].set_ylabel("VAF")

# Panel E: Wobble examples (stable with high variance)
wobble = traj[(traj["trajectory"] == "stable") & (traj["n_timepoints"] >= 3)].copy()
# Pick 5 with widest gap between first and last VAF
wobble["gap"] = (wobble["first_VAF"] - wobble["last_VAF"]).abs()
top_wobble = wobble.sort_values("gap", ascending=False).head(5)
axes[1,1].barh(range(5), top_wobble["gap"].values, color="gray")
axes[1,1].set_yticks(range(5))
axes[1,1].set_yticklabels([f"{r['variant_id'].split(' | ')[-1][:20]}\n({r['patient_id']})" for _, r in top_wobble.iterrows()], fontsize=7)
axes[1,1].set_xlabel("|first_VAF - last_VAF|")
axes[1,1].set_title("E) 'Stable' with largest VAF gap")

# Panel F: Noise sensitivity curve
noise_levels = [0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
flip_rates = []
for nl in noise_levels:
    noisy_last = stable_sample["last_VAF"].values + np.random.normal(0, nl, n_sample)
    noisy_last = np.clip(noisy_last, 0, 1)
    inc_f = (noisy_last > 1.5 * stable_sample["first_VAF"].values).sum()
    dec_f = (noisy_last < 0.5 * stable_sample["first_VAF"].values).sum()
    flip_rates.append((inc_f + dec_f) / n_sample * 100)

axes[1,2].plot(noise_levels, flip_rates, "go-", lw=2)
axes[1,2].set_xlabel("Noise standard deviation"); axes[1,2].set_ylabel("% trajectories reclassified")
axes[1,2].set_title("F) Stable -> flip under noise (n=%d)" % n_sample)

plt.tight_layout()
plt.savefig("stress_test_heuristic.png", dpi=150)
print("Saved stress_test_heuristic.png\nDone.")
