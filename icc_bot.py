#!/usr/bin/env python3
"""
ICC (Indication -> Correction -> Continuation) analysis bot.
ANALYSIS + ALERTS ONLY. It never places, buys or sells anything.

Pairs : USDJPY, XAUUSD (gold), NAS100, USDCHF, BTCUSD
Modes : SWING = daily bias, ICC on 4H      (hold days)
        DAY   = 4H/daily bias, ICC on 1H   (close same day) - no scalping
Alerts: ntfy.sh push (free, no signup) and/or Telegram, plus console. Heads-up 10 min before the candle close where
        the entry can trigger, then an ENTRY SIGNAL when the close confirms.

Data :  MetaTrader 5 terminal (default if the MetaTrader5 package is installed,
        Windows only) so levels match your broker; otherwise Yahoo Finance.
Setup:  pip install MetaTrader5 pandas numpy requests      (yfinance only for Yahoo)
        Open MT5, log in to your broker, keep it running.
Env  :  NTFY_TOPIC (your private topic name; free ntfy app on your phone)
        TELEGRAM_TOKEN, TELEGRAM_CHAT_ID (optional), TZ_NAME (e.g. Africa/Kampala)
        DATA_SOURCE=mt5|yahoo   MT5_SERVER_UTC_OFFSET (hours, optional override)
        MT5_SYMBOL_XAUUSD=GOLD  (override a broker symbol name, same for others)
        MT5_LOGIN / MT5_PASSWORD / MT5_SERVER / MT5_PATH (only if needed)
Run  :  python icc_bot.py            (loop)
        python icc_bot.py --headsup | --signal   (one-shot, for schedulers)
        python icc_bot.py --once     (single scan, prints setups)
        python icc_bot.py --test-alert
"""
import os, sys, time, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

UTC = dt.timezone.utc
LOCAL_TZ = ZoneInfo(os.getenv("TZ_NAME", "UTC"))
HEADS_UP_MIN = 10

INSTRUMENTS = {
    "USDJPY": dict(yf="USDJPY=X", sessions=["tokyo", "london", "newyork"], dec=3),
    "XAUUSD": dict(yf="GC=F",     sessions=["london", "newyork"],          dec=2),
    "NAS100": dict(yf="NQ=F",     sessions=["newyork"],                    dec=1),
    "USDCHF": dict(yf="USDCHF=X", sessions=["london", "newyork"],          dec=5),
    "BTCUSD": dict(yf="BTC-USD",  sessions=["london", "newyork"],          dec=1, weekend=True),
}

# (timezone, start (h, m), end (h, m)) in each session's LOCAL time, so DST is handled.
KILLZONE = {  # day trades: only the high-activity windows
    "tokyo":   ("Asia/Tokyo",       (9, 0), (12, 0)),
    "london":  ("Europe/London",    (8, 0), (11, 30)),
    "newyork": ("America/New_York", (8, 30), (11, 30)),
}
FULL = {      # swing trades: whole active session
    "tokyo":   ("Asia/Tokyo",       (9, 0), (15, 0)),
    "london":  ("Europe/London",    (8, 0), (16, 30)),
    "newyork": ("America/New_York", (8, 0), (16, 0)),
}


# ----------------------------------------------------------------- sessions
def in_window(sym, t, mode):
    """Return session name if time t (aware, UTC) is an allowed entry time."""
    cfg = INSTRUMENTS[sym]
    if t.weekday() >= 5 and not cfg.get("weekend"):
        return None
    table = KILLZONE if mode == "DAY" else FULL
    for s in cfg["sessions"]:
        tz, (h1, m1), (h2, m2) = table[s]
        lt = t.astimezone(ZoneInfo(tz))
        if lt.weekday() >= 5 and not cfg.get("weekend"):
            continue
        mins = lt.hour * 60 + lt.minute
        if h1 * 60 + m1 <= mins <= h2 * 60 + m2:
            return s
    return None


# --------------------------------------------------------------------- data
def _clean(df):
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df


def _closed_only(df, delta):
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + delta <= now]


def to_h4(h1):
    return h1.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


MT5_ALIASES = {
    "USDJPY": ["USDJPY"], "XAUUSD": ["XAUUSD", "GOLD"],
    "NAS100": ["NAS100", "US100", "USTEC", "NDX100"],
    "USDCHF": ["USDCHF"], "BTCUSD": ["BTCUSD", "BTCUSDT"],
}
_MT5 = {"ready": False, "offset": 0, "names": {}}
def _has_mt5():
    try:
        import MetaTrader5  # noqa: F401
        return True
    except ImportError:
        return False


SOURCE = os.getenv("DATA_SOURCE") or ("mt5" if _has_mt5() else "yahoo")


def _mt5_init():
    import MetaTrader5 as mt5
    if not _MT5["ready"]:
        kw = {}
        if os.getenv("MT5_PATH"):
            kw["path"] = os.environ["MT5_PATH"]
        if os.getenv("MT5_LOGIN"):
            kw.update(login=int(os.environ["MT5_LOGIN"]),
                      password=os.getenv("MT5_PASSWORD", ""), server=os.getenv("MT5_SERVER", ""))
        if not mt5.initialize(**kw):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        _MT5["ready"] = True
    return mt5


def _mt5_symbol(mt5, sym):
    if sym in _MT5["names"]:
        return _MT5["names"][sym]
    forced = os.getenv(f"MT5_SYMBOL_{sym}")
    name = forced
    if not name:
        for cand in MT5_ALIASES[sym]:
            found = [s.name for s in (mt5.symbols_get(f"*{cand}*") or [])]
            if found:
                name = cand if cand in found else min(found, key=len)
                break
    if not name or not mt5.symbol_select(name, True):
        raise RuntimeError(f"no broker symbol for {sym}; set MT5_SYMBOL_{sym}")
    INSTRUMENTS[sym]["dec"] = mt5.symbol_info(name).digits
    _MT5["names"][sym] = name
    return name


def _mt5_update_offset(mt5, name):
    """Broker server time is usually UTC+2/+3. Detect it from a fresh tick."""
    if os.getenv("MT5_SERVER_UTC_OFFSET"):
        _MT5["offset"] = float(os.environ["MT5_SERVER_UTC_OFFSET"])
        return
    tick = mt5.symbol_info_tick(name)
    if tick and tick.time:
        diff = tick.time - time.time()
        if abs(diff - round(diff / 3600) * 3600) < 300:   # fresh tick only
            _MT5["offset"] = round(diff / 3600)


def _mt5_rates(mt5, name, tf, count):
    r = mt5.copy_rates_from_pos(name, tf, 1, count)       # pos 1 = skip forming bar
    if r is None or len(r) == 0:
        raise RuntimeError(f"no MT5 data for {name}: {mt5.last_error()}")
    df = pd.DataFrame(r)
    df.index = pd.to_datetime(df["time"] - _MT5["offset"] * 3600, unit="s", utc=True)
    return df[["open", "high", "low", "close"]]


def load(sym):
    if SOURCE == "mt5":
        mt5 = _mt5_init()
        name = _mt5_symbol(mt5, sym)
        _mt5_update_offset(mt5, name)
        return (_mt5_rates(mt5, name, mt5.TIMEFRAME_H1, 2500),
                _mt5_rates(mt5, name, mt5.TIMEFRAME_H4, 1500),
                _mt5_rates(mt5, name, mt5.TIMEFRAME_D1, 500))
    import yfinance as yf
    tk = yf.Ticker(INSTRUMENTS[sym]["yf"])
    h1 = _clean(tk.history(period="120d", interval="1h", auto_adjust=False))
    d1 = _clean(tk.history(period="2y", interval="1d", auto_adjust=False))
    h1 = _closed_only(h1, pd.Timedelta(hours=1))
    h4 = _closed_only(to_h4(h1), pd.Timedelta(hours=4))
    d1 = _closed_only(d1, pd.Timedelta(days=1))
    return h1, h4, d1


# ---------------------------------------------------------------- structure
def atr(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def pivots(df, n=3):
    h, l, out = df.high.values, df.low.values, []
    for i in range(n, len(df) - n):
        if h[i] >= h[i - n:i + n + 1].max():
            out.append((i, "H", h[i]))
        if l[i] <= l[i - n:i + n + 1].min():
            out.append((i, "L", l[i]))
    return out


def bias(df, n=3):
    """+1 up, -1 down, 0 neutral (structure + EMA50)."""
    p = pivots(df, n)
    hs = [x[2] for x in p if x[1] == "H"][-2:]
    ls = [x[2] for x in p if x[1] == "L"][-2:]
    if len(hs) < 2 or len(ls) < 2:
        return 0
    ema, c = df.close.ewm(span=50).mean().iloc[-1], df.close.iloc[-1]
    if c > ema and ls[1] > ls[0]:
        return 1
    if c < ema and hs[1] < hs[0]:
        return -1
    return 0


# ---------------------------------------------------------------------- ICC
def _icc_long(df, n=3, buf=0.25):
    """
    Indication   : close breaks a swing high (break of structure up)
    Correction   : pullback of 30-79% of the impulse leg, origin never broken
    Continuation : close back above the pullback's last lower high
    Tries recent breaks newest-first and keeps the first valid ICC sequence.
    """
    piv = pivots(df, n)
    hs = [p for p in piv if p[1] == "H"]
    ls = [p for p in piv if p[1] == "L"]
    c, h, l, N = df.close.values, df.high.values, df.low.values, len(df)
    a = atr(df)

    breaks = set()
    for i, _, price in hs:
        for j in range(i + n + 1, N):
            if c[j] > price:
                breaks.add(j)
                break
    for j in sorted(breaks, reverse=True)[:8]:
        before = [p for p in ls if p[0] < j]
        if not before:
            continue
        L0 = before[-1][2]                               # origin of impulse
        k = j + int(np.argmax(h[j:]))                    # impulse high bar
        H1 = h[k]
        if k >= N - 1:
            continue                                     # still impulsing
        cp = k + 1 + int(np.argmin(l[k + 1:]))           # correction low bar
        C = l[cp]
        if C <= L0 or H1 <= L0:
            continue
        retr = (H1 - C) / (H1 - L0)
        if not 0.30 <= retr <= 0.79:
            continue
        th = [p for p in hs if p[0] > k and p[2] < H1]   # lower high inside pullback
        if not th:
            continue
        ti, _, trig = th[-1]
        broke = [m for m in range(ti + 1, N) if c[m] > trig]
        if broke and broke[0] < N - 1:
            continue                                     # already triggered earlier: missed
        status = "CONFIRMED" if broke else "PENDING"
        if status == "PENDING" and trig - c[-1] > 1.5 * a:
            continue                                     # too far from trigger
        entry = c[-1] if status == "CONFIRMED" else trig + 0.05 * a
        sl = C - buf * a
        risk = entry - sl
        if risk <= 0:
            continue
        tp1 = H1 if (H1 - entry) >= risk else entry + 1.5 * risk
        tp2 = max(H1 + 0.618 * (H1 - L0), entry + 2.5 * risk, tp1 + 0.5 * risk)
        return dict(status=status, trigger=trig, entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                    origin=L0, impulse_high=H1, corr_low=C, retrace=retr)
    return None


def icc(df, direction):
    if direction == 1:
        return _icc_long(df)
    f = pd.DataFrame({"open": -df.open, "high": -df.low, "low": -df.high, "close": -df.close})
    r = _icc_long(f)
    if not r:
        return None
    for k in ("trigger", "entry", "sl", "tp1", "tp2", "origin", "impulse_high", "corr_low"):
        r[k] = -r[k]
    return r


# ------------------------------------------------------------------ analyse
def analyse(sym, h1, h4, d1):
    out = []
    bd, b4 = bias(d1), bias(h4)
    plans = [
        ("SWING", h4, bd, pd.Timedelta(hours=4), "4H", "Daily"),
        ("DAY", h1, b4 if b4 != -bd else 0, pd.Timedelta(hours=1), "1H", "4H"),
    ]
    for mode, df, d, delta, tf, bias_tf in plans:
        if d == 0:
            continue
        r = icc(df, d)
        if not r:
            continue
        last_open = df.index[-1].to_pydatetime()
        r.update(symbol=sym, mode=mode, tf=tf, bias_tf=bias_tf,
                 side="BUY" if d == 1 else "SELL")
        if r["status"] == "CONFIRMED":
            r["entry_time"] = last_open + delta
        else:  # first in-session candle close that can trigger it
            t = last_open + 2 * delta
            for _ in range(96):
                if in_window(sym, t, mode):
                    break
                t += delta
            else:
                continue
            r["entry_time"] = t
        r["session"] = in_window(sym, r["entry_time"], mode)
        if r["status"] == "CONFIRMED" and not r["session"]:
            continue  # confirmed outside session rules: skip
        out.append(r)
    return out


# ------------------------------------------------------------------- alerts
def notify(msg):
    print(msg, flush=True)
    topic = os.getenv("NTFY_TOPIC")
    if topic:
        try:
            requests.post(f"https://ntfy.sh/{topic}", data=msg.encode("utf-8"),
                          headers={"Title": "ICC Bot", "Priority": "high"}, timeout=15)
        except Exception as e:
            print("ntfy error:", e)
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if tok and chat:
        try:
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={"chat_id": chat, "text": msg}, timeout=15)
        except Exception as e:
            print("telegram error:", e)


def fmt(s, kind):
    d = INSTRUMENTS[s["symbol"]]["dec"]
    p = lambda x: f"{x:.{d}f}"
    risk = abs(s["entry"] - s["sl"])
    rr = lambda tp: abs(tp - s["entry"]) / risk
    t = s["entry_time"].astimezone(LOCAL_TZ).strftime("%a %H:%M")
    head = {"HEADS_UP": f"⏰ HEADS-UP (10 min) - possible {s['side']} entry",
            "SIGNAL": f"✅ ENTRY SIGNAL - {s['side']} confirmed"}[kind]
    when = "Candle closes" if kind == "HEADS_UP" else "Confirmed at"
    trig = "close " + ("above " if s["side"] == "BUY" else "below ") + p(s["trigger"])
    return (f"{head}\n{s['symbol']} | {s['mode']} | ICC on {s['tf']} (bias {s['bias_tf']})\n"
            f"{when}: {t} ({s['session']} session)\n"
            + (f"Trigger: {trig}\n" if kind == "HEADS_UP" else "")
            + f"Entry: {p(s['entry'])}\nSL: {p(s['sl'])}\n"
            f"TP1: {p(s['tp1'])} ({rr(s['tp1']):.1f}R)\nTP2: {p(s['tp2'])} ({rr(s['tp2']):.1f}R)\n"
            f"Pullback: {s['retrace']*100:.0f}% of impulse\n"
            f"Setup is void if price takes out {p(s['corr_low'])} before entry.\n"
            f"Analysis only - not financial advice. You place the trade.")


def scan():
    found = []
    for sym in INSTRUMENTS:
        try:
            h1, h4, d1 = load(sym)
            found += analyse(sym, h1, h4, d1)
        except Exception as e:
            print(f"[{sym}] scan error: {e}", flush=True)
    return found


def main():
    if "--test-alert" in sys.argv:
        notify("ICC bot test alert: notifications are working.")
        return
    if "--headsup" in sys.argv or "--signal" in sys.argv:
        # One-shot modes for a scheduler (e.g. GitHub Actions): run at :50 and :02 each hour.
        now = dt.datetime.now(UTC)
        found = scan()
        for s in found:
            left = (s["entry_time"] - now).total_seconds()
            if "--headsup" in sys.argv and s["status"] == "PENDING" and 0 < left <= 15 * 60:
                notify(fmt(s, "HEADS_UP"))
            if "--signal" in sys.argv and s["status"] == "CONFIRMED" and -20 * 60 <= left <= 0:
                notify(fmt(s, "SIGNAL"))
        print(f"scan done, {len(found)} setups")
        return
    sent, setups, last_key = set(), [], None
    while True:
        now = dt.datetime.now(UTC)
        key = now.strftime("%Y%m%d%H")
        if now.minute >= 1 and key != last_key:       # scan once an hour, just after the close
            last_key = key
            setups = scan()
            for s in setups:
                if s["status"] == "CONFIRMED":
                    k = (s["symbol"], s["mode"], s["side"], round(s["trigger"], 4), "SIG")
                    if k not in sent:
                        sent.add(k)
                        notify(fmt(s, "SIGNAL"))
            if "--once" in sys.argv:
                for s in setups:
                    if s["status"] == "PENDING":
                        notify(fmt(s, "HEADS_UP"))
                if not setups:
                    print("No ICC setups right now.")
                return
        for s in setups:
            if s["status"] != "PENDING":
                continue
            left = (s["entry_time"] - now).total_seconds()
            k = (s["symbol"], s["mode"], s["side"], round(s["trigger"], 4), s["entry_time"])
            if 0 < left <= HEADS_UP_MIN * 60 and k not in sent:
                sent.add(k)
                notify(fmt(s, "HEADS_UP"))
        time.sleep(30)


if __name__ == "__main__":
    main()

