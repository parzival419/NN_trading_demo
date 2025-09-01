# =========================================================
# Walk-Forward LSTM vs Logistic baseline (V2)
#  - Adds stitched Buy&Hold overlay (same test windows)
#  - Prints per-fold win rate, avg trade P&L, avg hold
#  - Keeps dtype fixes: float32 features, int64 targets
# =========================================================

import warnings, math, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# Universe / target
train_tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]   # training universe
target_ticker = "AAPL"                              # evaluate/backtest on this one
sector_etf    = "XLK"                               # sector ETF context
start         = "2014-01-01"

# Features / labeling
window   = 60          # lookback days
horizon  = 1           # predict next-day direction

# Training (LSTM)
batch_sz = 64
epochs   = 60
lr       = 1e-3
weight_decay = 2e-4
early_stop_patience = 10

# Trading rules
base_threshold = 0.55   # replaced per-fold by val-selected
dead_zone      = 0.02   # buffer above threshold
fee            = 0.0005 # 5 bps per side
allow_short    = False  # keep off for now
target_ann_vol = 0.10   # vol targeting (annualized)

# Sentiment
enable_sentiment = True
sent_scorer      = "vader"    # "vader" by default
sent_fill_method = "zero"     # "zero" or "ffill"

# Walk-forward window lengths (trading days ~252/yr)
TR_DAYS = 252*3      # 3y train
VA_DAYS = 126        # 6m val
TE_DAYS = 126        # 6m test

# ---------------------------------------------------------
# Sentiment helpers (VADER default; robust to missing)
# ---------------------------------------------------------
def ensure_vader() -> bool:
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            SentimentIntensityAnalyzer()
            return True
        except LookupError:
            import nltk
            nltk.download("vader_lexicon")
            SentimentIntensityAnalyzer()
            return True
    except Exception:
        return False

def score_vader(texts):
    ok = ensure_vader()
    if not ok:
        return np.zeros(len(texts), dtype=float)
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t or "")["compound"] for t in texts], dtype=float)

def fetch_news_yf(ticker: str, start: str) -> pd.DataFrame:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []
    rows = []
    for it in items:
        ts = it.get("providerPublishTime")
        title = it.get("title")
        if ts and title:
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            rows.append({"published": dt, "title": str(title)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["published"] >= pd.to_datetime(start, utc=True)]
        df = df.sort_values("published")
    return df

def build_daily_sentiment(news_df: pd.DataFrame, scorer="vader", fill="zero") -> pd.DataFrame:
    idx_empty = pd.DatetimeIndex([], tz="UTC")
    blank = pd.DataFrame(index=idx_empty, columns=[
        "sent_mean","sent_std","sent_pos","sent_neg","sent_cnt","sent_ema3"
    ]).astype(float)
    if news_df is None or news_df.empty:
        return blank

    df = news_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["published"]):
        df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.dropna(subset=["published","title"])
    if df.empty: return blank

    texts = df["title"].astype(str).tolist()
    comp = score_vader(texts) if scorer.lower()=="vader" else score_vader(texts)

    pos = np.maximum(comp, 0.0); neg = np.maximum(-comp, 0.0)
    df["score"]=comp; df["pos"]=pos; df["neg"]=neg
    df["date"]=df["published"].dt.floor("D")

    g = df.groupby("date")
    daily = pd.DataFrame({
        "sent_mean": g["score"].mean(),
        "sent_std":  g["score"].std(),
        "sent_pos":  g["pos"].mean(),
        "sent_neg":  g["neg"].mean(),
        "sent_cnt":  g["score"].count().astype(float),
    })
    daily["sent_ema3"] = daily["sent_mean"].ewm(span=3, adjust=False).mean()
    daily = daily.shift(1)  # lag to avoid leakage
    if fill == "ffill":
        daily = daily.fillna(method="ffill").fillna(0.0)
    else:
        daily = daily.fillna(0.0)
    # make index tz-naive to match prices
    daily.index = pd.DatetimeIndex(daily.index.tz_convert(None))
    return daily

# ---------------------------------------------------------
# Data & features
# ---------------------------------------------------------
def dl_symbols(symbols, start):
    return yf.download(symbols, start=start, auto_adjust=True, group_by="ticker", progress=False)

def get_close_vol(raw, sym):
    df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
    df = df.rename(columns={c:c.lower() for c in df.columns})
    close = df["close"]
    vol   = df["volume"] if "volume" in df.columns else None
    return close.rename(f"{sym}_close"), (vol.rename(f"{sym}_vol") if vol is not None else None)

def build_features_for_ticker(base_close, base_vol, spy_close, vix_close, sector_close,
                              sent_daily: pd.DataFrame|None):
    close = base_close
    if base_vol is None:
        base_vol = pd.Series(0.0, index=close.index, name=close.name.replace("_close","_vol"))
    volume = base_vol

    # stock technicals
    ret1  = close.pct_change(1);  ret5  = close.pct_change(5);  ret10 = close.pct_change(10)
    std5  = ret1.rolling(5).std(); std10 = ret1.rolling(10).std()
    rsi   = RSIIndicator(close, window=14).rsi()
    sma5  = SMAIndicator(close, window=5).sma_indicator()
    sma20 = SMAIndicator(close, window=20).sma_indicator()
    sma_x = (sma5 - sma20) / sma20
    macd  = MACD(close); macd_line, macd_signal, macd_hist = macd.macd(), macd.macd_signal(), macd.macd_diff()
    vol_z = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std())

    # SPY context
    spy_r1 = spy_close.pct_change(1); spy_r5 = spy_close.pct_change(5); spy_r10 = spy_close.pct_change(10)
    spy_std5 = spy_r1.rolling(5).std(); spy_std10 = spy_r1.rolling(10).std()
    spy_pos = (spy_close - spy_close.rolling(20).mean()) / spy_close.rolling(20).mean()

    # VIX context
    vix_level = vix_close
    vix_chg1  = vix_close.pct_change(1); vix_chg5 = vix_close.pct_change(5)
    vix_z     = (vix_close - vix_close.rolling(20).mean()) / vix_close.rolling(20).std()

    # Sector context
    sec_r1 = sector_close.pct_change(1); sec_r5 = sector_close.pct_change(5); sec_r10 = sector_close.pct_change(10)
    sec_std5 = sec_r1.rolling(5).std(); sec_std10 = sec_r1.rolling(10).std()
    sec_pos = (sector_close - sector_close.rolling(20).mean()) / sector_close.rolling(20).mean()

    # Relative strength vs sector (20d)
    rel20 = (close.pct_change(20) - sector_close.pct_change(20)).rename("rel20")

    feat = pd.concat([
        # stock
        ret1.rename("ret1"), ret5.rename("ret5"), ret10.rename("ret10"),
        std5.rename("std5"), std10.rename("std10"),
        rsi.rename("rsi"), sma_x.rename("sma_cross"),
        macd_line.rename("macd"), macd_signal.rename("macd_sig"), macd_hist.rename("macd_hist"),
        vol_z.rename("volz"),
        # spy
        spy_r1.rename("spy_r1"), spy_r5.rename("spy_r5"), spy_r10.rename("spy_r10"),
        spy_std5.rename("spy_std5"), spy_std10.rename("spy_std10"), spy_pos.rename("spy_pos"),
        # vix
        vix_level.rename("vix"), vix_chg1.rename("vix_chg1"), vix_chg5.rename("vix_chg5"), vix_z.rename("vix_z"),
        # sector
        sec_r1.rename("sec_r1"), sec_r5.rename("sec_r5"), sec_r10.rename("sec_r10"),
        sec_std5.rename("sec_std5"), sec_std10.rename("sec_std10"), sec_pos.rename("sec_pos"),
        # relative strength
        rel20
    ], axis=1)

    # sentiment block
    sent_cols = ["sent_mean","sent_std","sent_pos","sent_neg","sent_cnt","sent_ema3"]
    if enable_sentiment:
        if sent_daily is not None and not sent_daily.empty:
            sent = sent_daily.reindex(feat.index, method=None)
            sent = sent.fillna(method="ffill").fillna(0.0) if sent_fill_method=="ffill" else sent.fillna(0.0)
            feat = pd.concat([feat, sent], axis=1)
        else:
            feat = pd.concat([feat, pd.DataFrame(0.0, index=feat.index, columns=sent_cols)], axis=1)

    label = (close.shift(-horizon) > close).astype(int).rename("label")
    df = pd.concat([feat, label], axis=1).dropna()
    return df, close

def to_sequences(df_features, y, win):
    F = df_features.values.astype(np.float32)  # ensure float32 early
    Y = y.values.astype(np.int64)
    idx = df_features.index
    X_list, y_list, t_idx = [], [], []
    for t in range(win, len(F)):
        X_list.append(F[t-win:t, :])
        y_list.append(Y[t])
        t_idx.append(idx[t])
    return np.stack(X_list), np.array(y_list), pd.DatetimeIndex(t_idx)

def equity_curve(r): return np.cumprod(1.0 + np.nan_to_num(r))
def max_drawdown(eq): peaks = np.maximum.accumulate(eq); return (eq/peaks - 1.0).min()
def perf_stats(r, periods=252):
    r = np.nan_to_num(r); eq = equity_curve(r)
    total = eq[-1]-1.0
    years = len(r)/periods if len(r) else np.nan
    cagr = (eq[-1])**(1/years)-1.0 if years and eq[-1]>0 else np.nan
    sharpe = (r.mean()/r.std())*np.sqrt(periods) if r.std()>0 else np.nan
    mdd = max_drawdown(eq)
    return dict(total=total, cagr=cagr, sharpe=sharpe, mdd=mdd)
def ann_sharpe(r, periods=252):
    r = np.nan_to_num(r)
    if r.std()==0: return float("nan")
    return (r.mean()/r.std()) * np.sqrt(periods)

# ---------------------------------------------------------
# Models (with dtype fixes)
# ---------------------------------------------------------
class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)  # DTYPE FIX
        self.y = torch.as_tensor(y, dtype=torch.long)     # DTYPE FIX
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=128, num_layers=2, bidirectional=False, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers>1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        out,_ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

def train_lstm(X_train, y_train, X_val, y_val, n_features):
    train_dl = DataLoader(SeqDS(X_train, y_train), batch_size=batch_sz, shuffle=True)
    val_dl   = DataLoader(SeqDS(X_val,   y_val),   batch_size=batch_sz)

    model = LSTMClassifier(n_features).to(device)

    # class-weighted loss
    neg = (y_train==0).sum(); pos = (y_train==1).sum()
    w0 = pos/(pos+neg+1e-9); w1 = neg/(pos+neg+1e-9)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([w0, w1], dtype=torch.float32, device=device)
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best, best_val, wait = None, math.inf, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device, dtype=torch.float32)  # DTYPE FIX
            yb = yb.to(device, dtype=torch.long)     # DTYPE FIX
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # validation
        model.eval(); vals=[]
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, dtype=torch.float32)
                yb = yb.to(device, dtype=torch.long)
                vals.append(loss_fn(model(xb), yb).item())
        v = float(np.mean(vals))
        if v < best_val - 1e-4:
            best_val = v
            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= early_stop_patience:
                break

    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return model

def predict_lstm(model, X):
    dl = DataLoader(SeqDS(X, np.zeros(len(X), dtype=np.int64)), batch_size=batch_sz)
    probs=[]
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device, dtype=torch.float32)  # DTYPE FIX
            p = torch.softmax(model(xb), dim=1)[:,1].cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)

def train_logit(X_train_last, y_train):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_last, y_train)
    return clf

# ---------------------------------------------------------
# Trading & per-trade stats
# ---------------------------------------------------------
def _trade_summary(idx, signal, strat_adj):
    """Compute win rate, avg trade pnl, avg hold days from daily series."""
    sig = pd.Series(signal, index=idx).astype(int)
    pnl = pd.Series(strat_adj, index=idx)
    # entries where we go from 0 -> 1
    entries = (sig==1) & (sig.shift(1, fill_value=0)==0)
    if entries.sum() == 0:
        return float("nan"), float("nan"), float("nan"), 0
    pos_id = entries.cumsum()
    pos_id = pos_id.where(sig==1, np.nan)

    trade_pnls = pnl.where(sig==1).groupby(pos_id).sum().dropna()
    trade_lens = sig.where(sig==1).groupby(pos_id).sum().dropna()

    win_rate = (trade_pnls>0).mean() if len(trade_pnls) else np.nan
    avg_pnl  = trade_pnls.mean() if len(trade_pnls) else np.nan
    avg_hold = trade_lens.mean() if len(trade_lens) else np.nan
    return float(win_rate), float(avg_pnl), float(avg_hold), int(len(trade_pnls))

def trade_and_stats(prob, idx, threshold, spy_pos_series, vix_z_series, ret_next, label_desc=""):
    regime = ((spy_pos_series>0) & (vix_z_series<0)).reindex(idx).fillna(False).values
    long_sig = ((prob >= (threshold + dead_zone)) & regime).astype(np.int8)
    if allow_short:
        short_sig = ((prob <= 1.0 - (threshold + dead_zone)) & regime).astype(np.int8)
        side = np.where(long_sig==1, 1, np.where(short_sig==1, -1, 0)).astype(np.int8)
    else:
        side = long_sig.copy()

    # vol sizing (on test index)
    ret_s = pd.Series(ret_next, index=idx)
    roll_vol = ret_s.rolling(20).std()
    ann_vol  = roll_vol * np.sqrt(252)
    size = (target_ann_vol / ann_vol).clip(0.0, 1.0).fillna(0.0).values

    signal = (side!=0).astype(np.int8)
    signed_side = side.astype(np.int8)
    strat = signed_side * size * ret_next

    sig_shift = np.roll(signal, 1); sig_shift[0]=0
    entries = (signal==1) & (sig_shift==0)
    exits   = (signal==0) & (sig_shift==1)

    strat_adj = strat.copy()
    strat_adj[entries] -= fee * size[entries]
    strat_adj[exits]   -= fee * size[exits]

    stats = perf_stats(strat_adj)
    win_rate, avg_pnl, avg_hold, n_closed = _trade_summary(idx, signal, strat_adj)
    n_trades = int(entries.sum()+exits.sum())
    extra = dict(win_rate=win_rate, avg_pnl=avg_pnl, avg_hold=avg_hold, n_closed=n_closed)
    return strat_adj, stats, n_trades, size, side, extra

# ---------------------------------------------------------
# Download and build features for all tickers
# ---------------------------------------------------------
extra = ["SPY", "^VIX", sector_etf]
universe = sorted(set(train_tickers + [target_ticker] + extra))
raw = dl_symbols(universe, start)

spy_close,_    = get_close_vol(raw, "SPY")
vix_close,_    = get_close_vol(raw, "^VIX")
sector_close,_ = get_close_vol(raw, sector_etf)

spy_pos_series = (spy_close - spy_close.rolling(20).mean())/spy_close.rolling(20).mean()
vix_z_series   = (vix_close - vix_close.rolling(20).mean())/vix_close.rolling(20).std()

ticker_df = {}
target_close_series = None

for sym in train_tickers + [target_ticker]:
    news = fetch_news_yf(sym, start) if enable_sentiment else None
    sent = build_daily_sentiment(news, scorer=sent_scorer, fill=sent_fill_method) if enable_sentiment else None

    c, v = get_close_vol(raw, sym)
    aligned = pd.concat([c, v, spy_close, vix_close, sector_close], axis=1).dropna()
    c = aligned[f"{sym}_close"]; v = aligned.get(f"{sym}_vol")
    sc = aligned["SPY_close"]; vx = aligned["^VIX_close"]; sec = aligned[f"{sector_etf}_close"]

    df, _ = build_features_for_ticker(c, v, sc, vx, sec, sent)
    ticker_df[sym] = df
    if sym == target_ticker:
        target_close_series = c

target_idx_full = ticker_df[target_ticker].index

# ---------------------------------------------------------
# Walk-forward folds (3y/6m/6m)
# ---------------------------------------------------------
folds = []
start_i = 0
while True:
    tr_end_i = start_i + TR_DAYS
    va_end_i = tr_end_i + VA_DAYS
    te_end_i = va_end_i + TE_DAYS
    if te_end_i + window >= len(target_idx_full):
        break
    tr_start, tr_end = target_idx_full[start_i], target_idx_full[tr_end_i]
    va_end           = target_idx_full[va_end_i]
    te_end           = target_idx_full[te_end_i]
    folds.append((tr_start, tr_end, va_end, te_end))
    start_i += TE_DAYS

print(f"Folds: {len(folds)}")

# ---------------------------------------------------------
# Train per fold, evaluate, stitch
# ---------------------------------------------------------
all_results = {"lstm":[], "logit":[]}
stitched = {"lstm":[], "logit":[], "bh":[]}
stitched_idx = {"lstm":[], "logit":[], "bh":[]}

for fi,(tr_s, tr_e, va_e, te_e) in enumerate(folds, 1):
    print(f"\n==== Fold {fi}: {tr_s.date()} .. {tr_e.date()} (train) | .. {va_e.date()} (val) | .. {te_e.date()} (test) ====")

    Xtr_list, ytr_list = [], []
    Xva_list, yva_list = [], []

    Xval_target = yval_target = idx_val_target = None
    Xtest_target = ytest_target = idx_test_target = None

    for sym in train_tickers + [target_ticker]:
        df = ticker_df[sym]
        X,y,idx = to_sequences(df.drop(columns=["label"]), df["label"], window)

        m_tr = (idx>tr_s) & (idx<=tr_e)
        m_va = (idx>tr_e) & (idx<=va_e)
        m_te = (idx>va_e) & (idx<=te_e)

        if sym in train_tickers:
            Xtr_list.append(X[m_tr]); ytr_list.append(y[m_tr])
            Xva_list.append(X[m_va]); yva_list.append(y[m_va])

        if sym == target_ticker:
            Xval_target, yval_target, idx_val_target = X[m_va], y[m_va], idx[m_va]
            Xtest_target, ytest_target, idx_test_target = X[m_te], y[m_te], idx[m_te]

    Xtr = np.concatenate(Xtr_list, axis=0); ytr = np.concatenate(ytr_list, axis=0)
    Xva = np.concatenate(Xva_list, axis=0); yva = np.concatenate(yva_list, axis=0)

    n_features = Xtr.shape[-1]

    # scale (fit on train only) and cast to float32
    scaler = StandardScaler().fit(Xtr.reshape(-1, n_features))
    def scale_block(X): return scaler.transform(X.reshape(-1, n_features)).reshape(X.shape).astype(np.float32)
    Xtr_s, Xva_s = scale_block(Xtr), scale_block(Xva)
    Xvt_s, Xtt_s = scale_block(Xval_target), scale_block(Xtest_target)

    # next-day returns & regime on val/test
    ret_next = target_close_series.pct_change(1).shift(-1)
    val_ret  = ret_next.reindex(idx_val_target).values
    test_ret = ret_next.reindex(idx_test_target).values

    # --------------- LSTM ---------------
    lstm = train_lstm(Xtr_s, ytr, Xva_s, yva, n_features)
    p_val_lstm  = predict_lstm(lstm, Xvt_s)
    p_test_lstm = predict_lstm(lstm, Xtt_s)

    # val threshold sweep (Sharpe)
    best_th = base_threshold; best_sh = -1e9
    for th in np.linspace(0.50, 0.70, 21):
        strat,_stats,_,_,_,_ = trade_and_stats(
            p_val_lstm, idx_val_target, th, spy_pos_series, vix_z_series, val_ret, "val")
        sh = ann_sharpe(strat)
        if not np.isnan(sh) and sh > best_sh: best_th, best_sh = float(th), sh

    strat_test, stats, n_trades, _, _, extra = trade_and_stats(
        p_test_lstm, idx_test_target, best_th, spy_pos_series, vix_z_series, test_ret, "test")
    all_results["lstm"].append(("Fold", fi, best_th, stats, n_trades, extra))
    stitched["lstm"].append(strat_test); stitched_idx["lstm"].append(idx_test_target)

    print(f"[LSTM] th={best_th:.2f} | Sharpe={stats['sharpe']:.2f} | Total={stats['total']:.2%} | "
          f"MDD={stats['mdd']:.2%} | trades={n_trades} | "
          f"win={extra['win_rate']:.2%} | avg_trade={extra['avg_pnl']:.3%} | avg_hold={extra['avg_hold']:.2f}d")

    # --------------- Logistic ---------------
    Xtr_last = Xtr_s[:, -1, :]
    Xvt_last = Xvt_s[:, -1, :]
    Xtt_last = Xtt_s[:, -1, :]

    logit = train_logit(Xtr_last, ytr)
    p_val_logit  = logit.predict_proba(Xvt_last)[:,1]
    p_test_logit = logit.predict_proba(Xtt_last)[:,1]

    best_th = base_threshold; best_sh = -1e9
    for th in np.linspace(0.50, 0.70, 21):
        strat,_stats,_,_,_,_ = trade_and_stats(
            p_val_logit, idx_val_target, th, spy_pos_series, vix_z_series, val_ret, "val")
        sh = ann_sharpe(strat)
        if not np.isnan(sh) and sh > best_sh: best_th, best_sh = float(th), sh

    strat_test, stats, n_trades, _, _, extra = trade_and_stats(
        p_test_logit, idx_test_target, best_th, spy_pos_series, vix_z_series, test_ret, "test")
    all_results["logit"].append(("Fold", fi, best_th, stats, n_trades, extra))
    stitched["logit"].append(strat_test); stitched_idx["logit"].append(idx_test_target)

    print(f"[LOGIT] th={best_th:.2f} | Sharpe={stats['sharpe']:.2f} | Total={stats['total']:.2%} | "
          f"MDD={stats['mdd']:.2%} | trades={n_trades} | "
          f"win={extra['win_rate']:.2%} | avg_trade={extra['avg_pnl']:.3%} | avg_hold={extra['avg_hold']:.2f}d")

    # --------------- Buy & Hold on SAME test window ---------------
    stitched["bh"].append(test_ret)            # raw daily returns for B&H
    stitched_idx["bh"].append(idx_test_target)

# ---------------------------------------------------------
# Stitched summaries
# ---------------------------------------------------------
def summarize_stitched(name):
    r = np.concatenate(stitched[name]) if len(stitched[name]) else np.array([])
    stats = perf_stats(r) if len(r) else {"total":np.nan,"cagr":np.nan,"sharpe":np.nan,"mdd":np.nan}
    print(f"\n==== {name.upper()} stitched test ====")
    print(f"Total:  {stats['total']:.2%}")
    print(f"CAGR:   {stats['cagr']:.2%}")
    print(f"Sharpe: {stats['sharpe']:.2f}")
    print(f"MDD:    {stats['mdd']:.2%}")
    return r, stats

r_lstm, s_lstm   = summarize_stitched("lstm")
r_logit, s_logit = summarize_stitched("logit")
r_bh,   s_bh     = summarize_stitched("bh")

# Plot stitched equity curves (overlay Buy&Hold)
plt.figure(figsize=(10,5))
if len(r_bh):    plt.plot(equity_curve(r_bh),   label="Buy & Hold (stitched)", color="gray")
if len(r_lstm):  plt.plot(equity_curve(r_lstm), label="LSTM (stitched)")
if len(r_logit): plt.plot(equity_curve(r_logit),label="Logistic (stitched)")
plt.title(f"{target_ticker} - Stitched Walk-Forward Equity")
plt.legend(); plt.tight_layout(); plt.show()
