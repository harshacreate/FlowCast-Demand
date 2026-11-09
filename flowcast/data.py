"""Synthetic chronological series for FlowCast-Demand — Hourly Trip-Demand Forecasting; no external data is used."""

from __future__ import annotations
import numpy as np

FEATURES = ["hour_sin", "hour_cos", "week_sin", "week_cos", "trend", "temperature", "rainfall", "event_flag", "lag_1", "lag_24", "lag_168", "rolling_mean_24"]
BASELINE_FEATURE = "lag_24"


UNIT = 'synthetic trips per hour'
def make_dataset(n=1200, seed=42):
    if n < 240:
        raise ValueError("n must be at least 240 to support a 168-hour history and holdout")
    rng = np.random.default_rng(seed)
    t=np.arange(168,168+n)
    full=np.arange(n+168)
    hour=full%24; dow=(full//24)%7
    weather=12+9*np.sin(2*np.pi*(full/24-5)/365)+rng.normal(0,2,n+168)
    rain=np.clip(rng.gamma(1.2,1.5,n+168),0,15)
    event=(rng.random(n+168)<.025).astype(float)
    noise=rng.normal(0,7,n+168)
    series=np.maximum(0,72+18*np.sin(2*np.pi*(hour-8)/24)+12*np.sin(2*np.pi*(dow)/7)+.018*full-0.55*weather-1.5*rain+28*event+noise)
    X=np.column_stack([np.sin(2*np.pi*hour[t]/24),np.cos(2*np.pi*hour[t]/24),np.sin(2*np.pi*dow[t]/7),np.cos(2*np.pi*dow[t]/7),t/(n+168),weather[t],rain[t],event[t],series[t-1],series[t-24],series[t-168],np.array([series[k-24:k].mean() for k in t])])
    y=series[t]
    if X.shape != (n, len(FEATURES)) or y.shape != (n,) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("generator produced invalid time-series rows")
    return X.astype(float), y.astype(float)
