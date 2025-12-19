import numpy as np
import csv
import gzip
import argparse
import time
from pathlib import Path

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _autocov(N: int, H: float) -> np.ndarray:
    k = np.arange(0, N, dtype=np.float64)
    return 0.5 * (
        np.abs(k + 1) ** (2 * H)
        - 2.0 * np.abs(k) ** (2 * H)
        + np.abs(k - 1) ** (2 * H)
    )

def _eigs_from_r(r: np.ndarray) -> np.ndarray:
    c = np.concatenate([r, [0.0], r[1:][::-1]])
    eigs = np.real(np.fft.fft(c))
    return np.maximum(eigs, 1e-12)

def _sample_fgn(eigs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    L = eigs.size
    N = L // 2

    W = np.empty(L, dtype=np.complex128)

    W[0] = rng.normal()
    W[N] = rng.normal()

    a = rng.normal(size=N - 1)
    b = rng.normal(size=N - 1)
    W[1:N] = (a + 1j * b) / np.sqrt(2.0)
    W[N + 1:] = np.conj(W[1:N][::-1])

    Y = np.sqrt(eigs) * W
    x = np.fft.ifft(Y).real
    return x[:N] / np.sqrt(L)

def sample_fbm(
    N: int,
    H: float,
    dt: float,
    rng: np.random.Generator,
    cache: dict,
) -> np.ndarray:
 
    key = (N, float(H))
    if key not in cache:
        r = _autocov(N, H)
        cache[key] = _eigs_from_r(r)
    eigs = cache[key]

    fgn = _sample_fgn(eigs, rng)  
    dB = (dt ** H) * fgn.astype(np.float64)

    B = np.empty(N + 1, dtype=np.float64)
    B[0] = 0.0
    np.cumsum(dB, out=B[1:])
    return B

def simulate_fou_from_fbm(
    B: np.ndarray,
    dt: float,
    theta: float,
    mu: float,
    base_sigma: float,
    rng: np.random.Generator,
    alpha: float = 0.7,
) -> np.ndarray:

    N = B.shape[0] - 1
    X = np.empty(N + 1, dtype=np.float64)
    X[0] = mu

    Z = B[:-1] - np.mean(B[:-1])
    vol = np.exp(alpha * Z)

    for i in range(N):
        eps = rng.standard_normal()
        sigma_i = base_sigma * vol[i]
        X[i + 1] = (
            X[i]
            + theta * (mu - X[i]) * dt
            + sigma_i * np.sqrt(dt) * eps
        )
    return X


def theta_func(H: float, dt: float, base_theta: float) -> float:

    rough = (1.0 - H) ** 1.5      
    short = max(1.0 - dt, 0.0) ** 0.5  

    val = base_theta * (0.15 + 2.5 * rough * short)
    return float(np.clip(val, 0.02, 0.6))  


def sigma_func(H: float, dt: float, base_sigma: float) -> float:

    rough = (1.0 - H) ** 1.2
    horizon = dt ** 0.7
    cross = (1.0 - H) ** 0.8 * horizon

    factor = 0.4 + 3.0 * rough + 1.5 * horizon + 2.0 * cross
    val = base_sigma * factor

    return float(np.clip(val, 0.05, 2.0)) 

def generate_dataset(
    out_csv: str,
    H_list,
    dt_list,
    T: float,
    dI: float,
    L: int,
    n_paths: int,
    per_dt: int,
    seed: int,
    base_theta: float,
    mu: float,
    base_sigma: float,
):


    rng = np.random.default_rng(seed)
    N = int(round(T / dI))
    cache = {}

    out_path = Path(out_csv)
    ensure_dir(out_path.parent)

    header = ["H", "dt"] + [f"X{i+1}" for i in range(L)] + ["Y", "X_t"]

    total_configs = len(H_list) * n_paths
    done = 0
    t0 = time.time()

    with gzip.open(out_path, "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow(header)

        for H in H_list:
            for _ in range(n_paths):

                B = sample_fbm(N, H, dI, rng, cache)

                for dt in dt_list:
                    k = int(round(dt / dI))
                    if k <= 0 or L + k >= len(B):
                        continue

                    theta_hd = theta_func(H, dt, base_theta)
                    sigma_hd = sigma_func(H, dt, base_sigma)

                    X = simulate_fou_from_fbm(B, dI, theta_hd, mu, sigma_hd, rng)

                    i_min = L
                    i_max = len(X) - k
                    if i_min >= i_max:
                        continue

                    idx_all = np.arange(i_min, i_max, dtype=int)
                    if per_dt > 0 and per_dt < len(idx_all):
                        idx = rng.choice(idx_all, size=per_dt, replace=False)
                    else:
                        idx = idx_all

                    for i in idx:
                        X_hist = X[i - L:i] - X[i - L]
                        Y = X[i + k] - X[i]
                        X_t = X[i]
                        row = [H, dt, *X_hist.astype(np.float32), float(Y), float(X_t)]
                        w.writerow(row)

                done += 1
                if done % 100 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"\r[gen] H_done_paths={done}/{total_configs} "
                        f"elapsed={elapsed/60:.2f} min",
                        end="",
                    )

    elapsed = time.time() - t0
    print(f"\n Done in {elapsed/60:.2f} min. Saved -> {out_path}")

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "fOU-on-fBm data generator."
        )
    )

    ap.add_argument("--out_csv", type=str, default="data_set/fbm.csv.gz")

    ap.add_argument(
        "--H_list",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    ap.add_argument(
        "--dt_list",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )

    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--dI", type=float, default=0.01)
    ap.add_argument("--L", type=int, default=60)

    ap.add_argument("--n_paths", type=int, default=200)
    ap.add_argument("--per_dt", type=int, default=120)

    ap.add_argument("--seed", type=int, default=2025)

    ap.add_argument("--theta", type=float, default=0.2)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.2)

    return ap.parse_args()

def main():
    a = parse_args()

    H_list = [float(x) for x in a.H_list.split(",") if x.strip()]
    dt_list = [float(x) for x in a.dt_list.split(",") if x.strip()]

    generate_dataset(
        out_csv=a.out_csv,
        H_list=H_list,
        dt_list=dt_list,
        T=a.T,
        dI=a.dI,
        L=a.L,
        n_paths=a.n_paths,
        per_dt=a.per_dt,
        seed=a.seed,
        base_theta=a.theta,
        mu=a.mu,
        base_sigma=a.sigma,
    )

if __name__ == "__main__":
    main()
