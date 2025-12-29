# ============================================
# Bermudan Put Option Pricing via LSM (Python)
# ============================================

import numpy as np
from scipy.stats import norm

# --------------------------
# 1. Black–Scholes European
# --------------------------

def blackscholes_price(K, T, S0, vol, r=0.0, q=0.0, callput='call'):
    """
    European option price in Black–Scholes model.
    """
    F = S0 * np.exp((r - q) * T)
    v = vol * np.sqrt(T)
    d1 = np.log(F / K) / v + 0.5 * v
    d2 = d1 - v
    try:
        opttype = {'call': 1, 'put': -1}[callput.lower()]
    except:
        raise ValueError('callput must be "call" or "put".')
    price = opttype * (F * norm.cdf(opttype * d1) - K * norm.cdf(opttype * d2)) * np.exp(-r * T)
    return price

# --------------------------
# 2. GBM path generator
# --------------------------

def generate_paths(S0, r, q, vol, T, n_steps, n_paths, seed=None):
    """
    Generate GBM paths under risk-neutral measure.
    S_{t+dt} = S_t * exp((r - q - 0.5*vol^2)*dt + vol*sqrt(dt)*Z)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    nudt = (r - q - 0.5 * vol**2) * dt
    sigsdt = vol * np.sqrt(dt)

    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0

    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        S[:, t] = S[:, t - 1] * np.exp(nudt + sigsdt * z)

    return S, dt

# --------------------------
# 3. Bermudan put via LSM
# --------------------------

def bermudan_put_lsm(S, K, r, dt, exercise_indices, degree=2):
    """
    Price a Bermudan put via Longstaff–Schwartz.

    Parameters
    ----------
    S : array, shape (n_paths, n_steps+1)
        Simulated price paths.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    dt : float
        Time step.
    exercise_indices : list of ints
        Time indices (0..n_steps) where early exercise is allowed.
        Must include final index (maturity).
    degree : int
        Degree of polynomial basis for continuation regression.

    Returns
    -------
    price : float
        Bermudan put price estimate at t=0.
    betas : dict
        Regression coefficients at each exercise index t (for analysis).
    """
    n_paths, n_steps_plus1 = S.shape
    n_steps = n_steps_plus1 - 1
    disc = np.exp(-r * dt)

    exercise_set = set(exercise_indices)

    # Intrinsic value at all times
    intrinsic = np.maximum(K - S, 0.0)

    # At maturity: payoff = intrinsic
    cashflow = intrinsic[:, -1].copy()
    exercise_time = np.full(n_paths, n_steps, dtype=int)

    betas = {}

    # Work backwards over exercise times (excluding maturity)
    # We skip t=0 in backward loop; price at t=0 is average discounted cashflow.
    exercise_indices_sorted = sorted(exercise_indices)
    for t in reversed(exercise_indices_sorted[:-1]):  # all exercise times except last
        St = S[:, t]
        intrinsic_t = intrinsic[:, t]

        # Only paths that are in the money and not yet exercised in the future
        alive = exercise_time > t
        itm = intrinsic_t > 0
        candidates = alive & itm

        if np.sum(candidates) <= degree:
            # Not enough points; treat continuation as zero
            betas[t] = np.zeros(degree + 1)
            continue

        # Discounted continuation value (from t+dt forward) to time t
        Y = cashflow[candidates] * disc

        # Polynomial basis: [1, S, S^2, ..., S^degree]
        S_itm = St[candidates]
        X = np.vstack([S_itm**d for d in range(degree + 1)]).T

        # Least-squares regression to estimate conditional expectation
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        betas[t] = beta

        # Predict continuation value for ALL paths at time t
        X_all = np.vstack([St**d for d in range(degree + 1)]).T
        C = X_all @ beta

        # Early exercise decision: exercise if intrinsic >= continuation
        exercise_now = (intrinsic_t >= C) & alive

        # Update cashflow and exercise time for exercising paths
        cashflow[exercise_now] = intrinsic_t[exercise_now]
        exercise_time[exercise_now] = t

    # Discount cashflows from exercise times to t=0
    discounted = cashflow * np.exp(-r * exercise_time * dt)
    price = discounted.mean()

    return price, betas

# --------------------------
# 4. Main experiment
# --------------------------

if __name__ == "__main__":
    # Model and contract parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    q = 0.0
    vol = 0.2
    T = 1.0

    # Simulation parameters
    n_steps = 50           # time discretization
    n_paths = 100_000      # Monte Carlo paths

    # Bermudan exercise dates: quarterly (including maturity)
    # Map exercise times to indices on the time grid.
    exercise_times = [0.25, 0.5, 0.75, 1.0]
    time_grid = np.linspace(0, T, n_steps + 1)
    exercise_indices = [np.argmin(np.abs(time_grid - t)) for t in exercise_times]

    # Generate paths under risk-neutral measure
    S, dt = generate_paths(S0, r, q, vol, T, n_steps, n_paths, seed=42)

    # Price Bermudan put with different polynomial degrees (as regression methods)
    for degree in [1, 2, 3]:
        berm_price, betas = bermudan_put_lsm(
            S=S,
            K=K,
            r=r,
            dt=dt,
            exercise_indices=exercise_indices,
            degree=degree
        )
        print(f"Bermudan put (degree={degree}): {berm_price:.4f}")

    # Compare to European put (no early exercise)
    euro_put = blackscholes_price(K=K, T=T, S0=S0, vol=vol, r=r, q=q, callput='put')
    print(f"European put (Black–Scholes): {euro_put:.4f}")
