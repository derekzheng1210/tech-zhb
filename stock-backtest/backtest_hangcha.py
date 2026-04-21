# -*- coding: utf-8 -*-
"""
杭叉集团(603298.SH) 技术指标回测脚本
回测区间：2024-04-17 ~ 2026-04-17（约5年）
策略：8种常见技术指标买卖信号
"""

import warnings
warnings.filterwarnings('ignore')

import requests
import json as _json
import pandas as pd
import numpy as np
import time
from datetime import datetime


def _fetch_from_tencent(symbol, start_date, end_date, headers):
    """腾讯日线接口（主接口）：支持指定日期范围，前复权"""
    prefix = "sh" if symbol.startswith("6") else "sz"
    tx_symbol = f"{prefix}{symbol}"
    sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={tx_symbol},day,{sd},{ed},800,qfq")

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != 0 or not data.get("data"):
        raise ValueError("腾讯接口返回异常")

    day_data = data["data"].get(tx_symbol, {})
    klines = day_data.get("qfqday") or day_data.get("day", [])
    if not klines:
        raise ValueError("腾讯接口返回空数据")

    # 格式: [日期, 开盘, 收盘, 最高, 最低, 成交量]
    records = []
    for k in klines:
        records.append({
            "date": k[0],
            "open": k[1],
            "close": k[2],
            "high": k[3],
            "low": k[4],
            "volume": k[5],
        })
    return pd.DataFrame(records)


def _fetch_from_sina(symbol, start_date, end_date, headers):
    """新浪日线接口（备用接口）：返回最近N条，需按日期过滤"""
    prefix = "sh" if symbol.startswith("6") else "sz"
    sina_symbol = f"{prefix}{symbol}"

    url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={sina_symbol}&scale=240&ma=no&datalen=1200")

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = _json.loads(resp.text)

    if not data:
        raise ValueError("新浪接口返回空数据")

    # 格式: {day, open, high, low, close, volume}
    records = []
    for k in data:
        records.append({
            "date": k["day"],
            "open": k["open"],
            "close": k["close"],
            "high": k["high"],
            "low": k["low"],
            "volume": k["volume"],
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    # 按日期范围过滤
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    df = df[(df["date"] >= sd) & (df["date"] <= ed)].reset_index(drop=True)
    return df


def fetch_data(symbol="000039", start_date="20240417", end_date="20260417", max_retries=3):
    """获取A股日线前复权数据（腾讯为主，新浪兜底），带重试"""
    last_error = None
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    sources = [
        ("腾讯", _fetch_from_tencent),
        ("新浪", _fetch_from_sina),
    ]

    for src_name, fetch_func in sources:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[数据获取-{src_name}] 第 {attempt}/{max_retries} 次下载 {symbol} 日线数据 ({start_date} ~ {end_date})...")

                df = fetch_func(symbol, start_date, end_date, headers)

                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)

                # 强制转数值，防止偶发返回异常字符串
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.dropna(subset=['date', 'open', 'close', 'high', 'low']).reset_index(drop=True)

                if df.empty:
                    raise ValueError(f"股票 {symbol} 清洗后无有效数据")

                if len(df) < 2:
                    raise ValueError(f"股票 {symbol} 有效交易日不足 2 天，无法回测")

                print(f"[数据获取-{src_name}] 完成，共 {len(df)} 个交易日，"
                      f"区间 {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
                return df

            except Exception as e:
                last_error = e
                print(f"[数据获取-{src_name}] 第 {attempt} 次失败: {e}")
                if attempt < max_retries:
                    time.sleep(attempt)

        print(f"[数据获取] {src_name}接口全部失败，尝试下一个数据源...")

    raise RuntimeError(f"所有数据源均失败，最后错误: {last_error}")


# ============================================================
# 2. 技术指标计算
# ============================================================
def calc_ma(df, short, long):
    """双均线"""
    df[f'ma{short}'] = df['close'].rolling(short).mean()
    df[f'ma{long}'] = df['close'].rolling(long).mean()
    return df

def calc_macd(df, fast=12, slow=26, signal=9):
    """MACD"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['dif'] = ema_fast - ema_slow
    df['dea'] = df['dif'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = 2 * (df['dif'] - df['dea'])
    return df

def calc_rsi(df, period=14):
    """RSI"""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - 100 / (1 + rs)
    return df

def calc_kdj(df, n=9, m1=3, m2=3):
    """KDJ"""
    low_n = df['low'].rolling(n).min()
    high_n = df['high'].rolling(n).max()
    rsv = (df['close'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    df['k'] = rsv.ewm(com=m1-1, adjust=False).mean()
    df['d'] = df['k'].ewm(com=m2-1, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    return df

def calc_boll(df, period=20, num_std=2):
    """布林带"""
    df['boll_mid'] = df['close'].rolling(period).mean()
    rolling_std = df['close'].rolling(period).std()
    df['boll_upper'] = df['boll_mid'] + num_std * rolling_std
    df['boll_lower'] = df['boll_mid'] - num_std * rolling_std
    return df


# ============================================================
# 3. 信号生成
# ============================================================
def signal_ma_cross(df, short, long):
    """双均线交叉信号：金叉买入，死叉卖出"""
    df = calc_ma(df.copy(), short, long)
    ma_s = f'ma{short}'
    ma_l = f'ma{long}'
    signal = pd.Series(0, index=df.index)
    # 金叉：短均线从下方穿过长均线
    signal[(df[ma_s] > df[ma_l]) & (df[ma_s].shift(1) <= df[ma_l].shift(1))] = 1
    # 死叉：短均线从上方穿过长均线
    signal[(df[ma_s] < df[ma_l]) & (df[ma_s].shift(1) >= df[ma_l].shift(1))] = -1
    return signal

def signal_macd(df):
    """MACD信号：DIF上穿DEA买入，下穿卖出"""
    df = calc_macd(df.copy())
    signal = pd.Series(0, index=df.index)
    signal[(df['dif'] > df['dea']) & (df['dif'].shift(1) <= df['dea'].shift(1))] = 1
    signal[(df['dif'] < df['dea']) & (df['dif'].shift(1) >= df['dea'].shift(1))] = -1
    return signal

def signal_rsi(df, period=14, oversold=30, overbought=70):
    """RSI信号：超卖区买入，超买区卖出"""
    df = calc_rsi(df.copy(), period)
    signal = pd.Series(0, index=df.index)
    # RSI从下方穿越超卖线 → 买入
    signal[(df['rsi'] > oversold) & (df['rsi'].shift(1) <= oversold)] = 1
    # RSI从上方穿越超买线 → 卖出
    signal[(df['rsi'] < overbought) & (df['rsi'].shift(1) >= overbought)] = -1
    return signal

def signal_kdj(df):
    """KDJ信号：K线上穿D线买入，下穿卖出"""
    df = calc_kdj(df.copy())
    signal = pd.Series(0, index=df.index)
    signal[(df['k'] > df['d']) & (df['k'].shift(1) <= df['d'].shift(1))] = 1
    signal[(df['k'] < df['d']) & (df['k'].shift(1) >= df['d'].shift(1))] = -1
    return signal

def signal_boll(df):
    """布林带信号：触下轨买入，触上轨卖出"""
    df = calc_boll(df.copy())
    signal = pd.Series(0, index=df.index)
    # 价格从下方突破下轨后回到下轨之上 → 买入
    signal[(df['close'] > df['boll_lower']) & (df['close'].shift(1) <= df['boll_lower'].shift(1))] = 1
    # 价格从上方跌破上轨后回到上轨之下 → 卖出
    signal[(df['close'] < df['boll_upper']) & (df['close'].shift(1) >= df['boll_upper'].shift(1))] = -1
    return signal

def signal_macd_rsi(df):
    """MACD+RSI组合信号"""
    df = calc_macd(df.copy())
    df = calc_rsi(df)
    signal = pd.Series(0, index=df.index)
    # MACD金叉 且 RSI < 50 → 买入
    macd_golden = (df['dif'] > df['dea']) & (df['dif'].shift(1) <= df['dea'].shift(1))
    signal[macd_golden & (df['rsi'] < 50)] = 1
    # MACD死叉 且 RSI > 50 → 卖出
    macd_dead = (df['dif'] < df['dea']) & (df['dif'].shift(1) >= df['dea'].shift(1))
    signal[macd_dead & (df['rsi'] > 50)] = -1
    return signal


# ============================================================
# 4. 回测引擎
# ============================================================
def backtest(df, signal, strategy_name, initial_capital=1_000_000):
    """
    回测引擎
    - 全仓进出
    - 信号次日开盘价执行
    - 佣金：万5（双边），印花税：千1（卖出）
    """
    if df is None or df.empty:
        raise ValueError(f"{strategy_name} 回测失败: 数据为空")

    if len(df) < 2:
        raise ValueError(f"{strategy_name} 回测失败: 交易日不足 2 天")

    if signal is None or len(signal) != len(df):
        raise ValueError(f"{strategy_name} 回测失败: 信号长度与行情数据不一致")

    commission_rate = 0.0005
    stamp_tax_rate = 0.001

    cash = initial_capital
    shares = 0
    position = 0
    trades = []

    dates = df['date'].values
    opens = df['open'].values
    closes = df['close'].values

    entry_price = 0
    entry_date = None
    portfolio_value = []

    for i in range(len(df)):
        if position == 1:
            portfolio_value.append(cash + shares * closes[i])
        else:
            portfolio_value.append(cash)

        if i == 0:
            continue

        prev_signal = signal.iloc[i - 1]

        if prev_signal == 1 and position == 0:
            buy_price = opens[i]
            buy_cost = buy_price * (1 + commission_rate)
            shares = int(cash / buy_cost / 100) * 100
            if shares > 0:
                cost = shares * buy_price * (1 + commission_rate)
                cash -= cost
                position = 1
                entry_price = buy_price
                entry_date = dates[i]

        elif prev_signal == -1 and position == 1:
            sell_price = opens[i]
            revenue = shares * sell_price
            sell_cost = revenue * (commission_rate + stamp_tax_rate)
            cash += revenue - sell_cost
            pnl_pct = (sell_price - entry_price) / entry_price
            trades.append({
                'entry_date': pd.Timestamp(entry_date).strftime('%Y-%m-%d'),
                'exit_date': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
                'entry_price': round(float(entry_price), 2),
                'exit_price': round(float(sell_price), 2),
                'pnl_pct': round(float(pnl_pct) * 100, 2),
                'shares': int(shares)
            })
            shares = 0
            position = 0

    final_value = cash + shares * closes[-1] if position == 1 else cash

    if position == 1:
        pnl_pct = (closes[-1] - entry_price) / entry_price
        trades.append({
            'entry_date': pd.Timestamp(entry_date).strftime('%Y-%m-%d'),
            'exit_date': pd.Timestamp(dates[-1]).strftime('%Y-%m-%d') + '(持仓中)',
            'entry_price': round(float(entry_price), 2),
            'exit_price': round(float(closes[-1]), 2),
            'pnl_pct': round(float(pnl_pct) * 100, 2),
            'shares': int(shares)
        })

    portfolio_value[-1] = final_value
    portfolio_series = pd.Series(portfolio_value, index=df['date'])

    total_return = (final_value - initial_capital) / initial_capital
    n_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    cummax = portfolio_series.cummax()
    drawdown = (portfolio_series - cummax) / cummax
    max_drawdown = drawdown.min()

    daily_returns = portfolio_series.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() * 252 - 0.025) / (daily_returns.std() * np.sqrt(252))
    else:
        sharpe = 0

    n_trades = len(trades)
    wins = [t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]
    losses = [t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0]

    n_wins = len(wins)
    win_rate = n_wins / n_trades if n_trades > 0 else 0

    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_loss_ratio = round(float(avg_win / avg_loss), 2) if avg_loss > 0 else '-'

    result = {
        'strategy': strategy_name,
        'total_return': round(float(total_return) * 100, 2),
        'annual_return': round(float(annual_return) * 100, 2),
        'max_drawdown': round(float(max_drawdown) * 100, 2),
        'sharpe': round(float(sharpe), 2),
        'win_rate': round(float(win_rate) * 100, 1),
        'n_trades': int(n_trades),
        'avg_win': round(float(avg_win), 2),
        'avg_loss': round(float(avg_loss), 2),
        'profit_loss_ratio': profit_loss_ratio if isinstance(profit_loss_ratio, str) else round(float(profit_loss_ratio), 2),
        'final_value': round(float(final_value), 0),
        'trades': trades,
        'portfolio_series': portfolio_series
    }
    return result
# ============================================================
# 5. 基准（买入持有）
# ============================================================
def buy_and_hold(df, initial_capital=1_000_000):
    """基准策略：首日开盘买入，持有到底"""
    if df is None or df.empty:
        raise ValueError("基准策略失败: 数据为空")

    if len(df) < 1:
        raise ValueError("基准策略失败: 无可用交易日")

    commission_rate = 0.0005
    buy_price = df['open'].iloc[0]
    buy_cost = buy_price * (1 + commission_rate)
    shares = int(initial_capital / buy_cost / 100) * 100
    cost = shares * buy_price * (1 + commission_rate)
    cash = initial_capital - cost
    final_value = cash + shares * df['close'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    n_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    portfolio_value = cash + shares * df['close']
    portfolio_series = pd.Series(portfolio_value.values, index=df['date'])
    cummax = portfolio_series.cummax()
    drawdown = (portfolio_series - cummax) / cummax
    max_drawdown = drawdown.min()
    daily_returns = portfolio_series.pct_change().dropna()
    sharpe = (daily_returns.mean() * 252 - 0.025) / (daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 1 and daily_returns.std() > 0 else 0

    return {
        'strategy': '买入持有(基准)',
        'total_return': round(float(total_return) * 100, 2),
        'annual_return': round(float(annual_return) * 100, 2),
        'max_drawdown': round(float(max_drawdown) * 100, 2),
        'sharpe': round(float(sharpe), 2),
        'win_rate': '-',
        'n_trades': 0,
        'avg_win': '-',
        'avg_loss': '-',
        'profit_loss_ratio': '-',
        'final_value': round(float(final_value), 0),
        'portfolio_series': {
            pd.Timestamp(k).strftime('%Y-%m-%d'): float(v)
            for k, v in portfolio_series.items()
        }
    }
# ============================================================
# 6. 主程序
# ============================================================
def main():
    # 获取数据
    df = fetch_data("000039", "20240417", "20260417")

    # 定义策略
    strategies = [
        ("MA5/MA20 均线交叉", lambda d: signal_ma_cross(d, 5, 20)),
        ("MA10/MA30 均线交叉", lambda d: signal_ma_cross(d, 10, 30)),
        ("MA20/MA60 均线交叉", lambda d: signal_ma_cross(d, 20, 60)),
        ("MACD", lambda d: signal_macd(d)),
        ("RSI(14)", lambda d: signal_rsi(d)),
        ("KDJ", lambda d: signal_kdj(d)),
        ("布林带(BOLL)", lambda d: signal_boll(d)),
        ("MACD+RSI 组合", lambda d: signal_macd_rsi(d)),
    ]

    results = []
    all_trades = {}

    print("\n" + "=" * 70)
    print("  中集集团(000039) 技术指标回测  |  2024-04 ~ 2026-04")
    print("=" * 70)

    for name, signal_func in strategies:
        print(f"\n[回测] {name} ...")
        sig = signal_func(df)
        result = backtest(df, sig, name)
        results.append(result)
        all_trades[name] = result['trades']
        print(f"  累计收益: {result['total_return']}%  |  夏普: {result['sharpe']}  |  "
              f"胜率: {result['win_rate']}%  |  交易次数: {result['n_trades']}")

    # 基准
    benchmark = buy_and_hold(df)
    print(f"\n[基准] 买入持有: 累计收益 {benchmark['total_return']}%  |  夏普: {benchmark['sharpe']}")

    # ============================================================
    # 7. 结果汇总表
    # ============================================================
    print("\n\n" + "=" * 120)
    print("  综合回测结果对比")
    print("=" * 120)

    header = f"{'策略':<20} {'累计收益%':>10} {'年化收益%':>10} {'最大回撤%':>10} {'夏普比率':>8} {'胜率%':>8} {'交易次数':>8} {'盈亏比':>8} {'超额收益%':>10}"
    print(header)
    print("-" * 120)

    # 基准行
    print(f"{'买入持有(基准)':<18} {benchmark['total_return']:>10} {benchmark['annual_return']:>10} "
          f"{benchmark['max_drawdown']:>10} {benchmark['sharpe']:>8} {'-':>8} {0:>8} {'-':>8} {'-':>10}")
    print("-" * 120)

    for r in results:
        excess = round(r['total_return'] - benchmark['total_return'], 2)
        print(f"{r['strategy']:<18} {r['total_return']:>10} {r['annual_return']:>10} "
              f"{r['max_drawdown']:>10} {r['sharpe']:>8} {r['win_rate']:>8} {r['n_trades']:>8} "
              f"{r['profit_loss_ratio']:>8} {excess:>10}")

    # ============================================================
    # 8. 排名
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  策略排名（按夏普比率）")
    print("=" * 70)
    sorted_results = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        excess = round(r['total_return'] - benchmark['total_return'], 2)
        print(f"  #{i}  {r['strategy']:<20}  夏普={r['sharpe']:<8}  累计收益={r['total_return']}%  超额={excess}%")

    print("\n" + "=" * 70)
    print("  策略排名（按累计收益率）")
    print("=" * 70)
    sorted_by_return = sorted(results, key=lambda x: x['total_return'], reverse=True)
    for i, r in enumerate(sorted_by_return, 1):
        excess = round(r['total_return'] - benchmark['total_return'], 2)
        print(f"  #{i}  {r['strategy']:<20}  累计收益={r['total_return']}%  超额={excess}%  最大回撤={r['max_drawdown']}%")

    # ============================================================
    # 9. 最优策略详细交易记录
    # ============================================================
    best = sorted_results[0]
    print(f"\n\n{'=' * 70}")
    print(f"  最优策略【{best['strategy']}】交易明细")
    print(f"{'=' * 70}")
    print(f"  {'序号':<4} {'买入日期':<12} {'卖出日期':<16} {'买入价':>8} {'卖出价':>8} {'盈亏%':>8}")
    print(f"  {'-' * 60}")
    for i, t in enumerate(best['trades'], 1):
        print(f"  {i:<4} {t['entry_date']:<12} {t['exit_date']:<16} {t['entry_price']:>8} "
              f"{t['exit_price']:>8} {t['pnl_pct']:>8}")

    # ============================================================
    # 10. 输出结构化数据供报告使用
    # ============================================================
    print("\n\n===JSON_START===")
    import json
    output = {
        'benchmark': {k: v for k, v in benchmark.items()},
        'strategies': [{k: v for k, v in r.items() if k not in ('trades', 'portfolio_series')} for r in results],
        'best_by_sharpe': best['strategy'],
        'best_by_return': sorted_by_return[0]['strategy'],
        'trades_detail': {best['strategy']: best['trades']},
        'data_range': f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}",
        'total_trading_days': len(df)
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("===JSON_END===")


if __name__ == '__main__':
    main()
