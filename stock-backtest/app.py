from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from backtest_hangcha import (
    fetch_data, backtest, buy_and_hold,
    signal_ma_cross, signal_macd, signal_rsi,
    signal_kdj, signal_boll, signal_macd_rsi
)
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__, static_folder='', static_url_path='')
CORS(app)


def series_to_dict(series):
    return {
        pd.Timestamp(k).strftime('%Y-%m-%d'): float(v)
        for k, v in series.items()
    }


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    try:
        data = request.get_json(silent=True) or {}

        stock_code = str(data.get('stock_code', '603298')).strip()
        start_date = str(data.get('start_date', '20240417')).strip()
        end_date = str(data.get('end_date', '20260417')).strip()

        if not stock_code:
            return jsonify({'error': '股票代码不能为空'}), 400

        if len(start_date) != 8 or len(end_date) != 8:
            return jsonify({'error': '日期格式必须为 YYYYMMDD'}), 400

        df = fetch_data(stock_code, start_date, end_date)

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
        strategy_errors = {}

        for name, signal_func in strategies:
            try:
                sig = signal_func(df.copy())
                result = backtest(df.copy(), sig, name)

                if 'portfolio_series' in result:
                    result['portfolio_series'] = series_to_dict(result['portfolio_series'])

                results.append(result)

            except Exception as e:
                strategy_errors[name] = str(e)
                print(f"[策略失败] {name}: {e}")
                traceback.print_exc()

        if not results:
            return jsonify({
                'error': '所有策略执行失败',
                'detail': strategy_errors
            }), 500

        benchmark = buy_and_hold(df.copy())

        sorted_results = sorted(results, key=lambda x: x['sharpe'], reverse=True)
        sorted_by_return = sorted(results, key=lambda x: x['total_return'], reverse=True)

        response = {
            'benchmark': benchmark,
            'strategies': results,
            'best_by_sharpe': sorted_results[0]['strategy'],
            'best_by_return': sorted_by_return[0]['strategy'],
            'trades_detail': {
                sorted_results[0]['strategy']: sorted_results[0]['trades']
            },
            'data_range': f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}",
            'total_trading_days': len(df),
            'strategy_errors': strategy_errors
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': f'后端执行失败: {str(e)}'
        }), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)