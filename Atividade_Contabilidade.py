import yfinance as yf
import pandas as pd

TICKERS = ["PETR4.SA", "AZUL4.SA", "VALE3.SA"]
START_DATE = "2021-01-02"
END_DATE = None
TOTAL_CAPITAL = 100000.0

WEIGHTS = {TICKERS[0]: 4.0, TICKERS[1]: 3.0, TICKERS[2]: 3.0}


def month_key(dt):
    return (dt.year, dt.month)

def mean(xs):
    n = len(xs)
    return sum(xs)/n if n else 0.0

def stddev(xs, sample=True):
    n = len(xs)
    if n <= 1:
        return 0.0
    m = mean(xs)
    s2 = sum((x - m) ** 2 for x in xs)
    denom = (n - 1) if sample else n
    return (s2 / denom) ** 0.5

def variance(xs, sample=True):
    n = len(xs)
    if n <= 1:
        return 0.0
    m = mean(xs)
    s2 = sum((x - m) ** 2 for x in xs)
    denom = (n - 1) if sample else n
    return s2 / denom

def covariance(xs, ys, sample=True):
    n = min(len(xs), len(ys))
    if n <= 1:
        return 0.0
    mx, my = mean(xs[:n]), mean(ys[:n])
    s = 0.0
    for i in range(n):
        s += (xs[i] - mx) * (ys[i] - my)
    denom = (n - 1) if sample else n
    return s / denom

def correlation(xs, ys):
    n = min(len(xs), len(ys))
    if n <= 1:
        return 0.0
    sx, sy = stddev(xs), stddev(ys)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return covariance(xs, ys) / (sx * sy)

def coefficient_of_variation(mu, sd):
    return (sd / abs(mu)) if mu != 0 else float("inf")

def cumulative_return(returns):
    acc = 1.0
    for r in returns:
        acc *= (1 + r)
    return acc - 1.0

def align_return_series(returns_by_ticker):
    month_sets = [set(months) for months, _ in returns_by_ticker.values()]
    common = set.intersection(*month_sets) if month_sets else set()
    common_sorted = sorted(common)
    aligned = {}
    for tkr, (months, rets) in returns_by_ticker.items():
        idx = {months[i]: i for i in range(len(months))}
        aligned[tkr] = [rets[idx[m]] for m in common_sorted]
    return aligned, common_sorted

def normalize_weights(weights_dict, tickers):
    s = sum(weights_dict.get(t, 0.0) for t in tickers)
    return {t: (weights_dict.get(t, 0.0) / s if s else 0.0) for t in tickers}

def compute_portfolio_monthly_returns(aligned_returns, weights):
    if not aligned_returns:
        return []
    any_ticker = next(iter(aligned_returns))
    n = len(aligned_returns[any_ticker])
    rp = []
    for i in range(n):
        s = 0.0
        for t, series in aligned_returns.items():
            s += weights.get(t, 0.0) * series[i]
        rp.append(s)
    return rp

def fetch_adj_close_monthly_end(ticker, start_date, end_date=None):
    data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False)
    dates = list(data.index)
    adj = list(data["Adj Close"]) if "Adj Close" in data.columns else []
    if not dates or len(dates) != len(adj):
        return [], []
    by_month = {}
    for dt, px in zip(dates, adj):
        if px is None:
            continue
        mk = month_key(dt.to_pydatetime())
        by_month[mk] = px  # último preço do mês
    months_sorted = sorted(by_month.keys())
    prices = [by_month[m] for m in months_sorted]
    return months_sorted, prices

def compute_monthly_returns_from_prices(prices):
    rets = []
    for i in range(1, len(prices)):
        p0, p1 = prices[i - 1], prices[i]
        rets.append((p1 / p0) - 1 if p0 else 0.0)
    return rets

def CAPM(aligned_returns, common_months, start_date, end_date=None, rf=None):
    if rf is None:
        rf_anual = 0.05
        rf = (1 + rf_anual) ** (1/12) - 1

    lista_capm = {}
    months_mkt, prices_mkt = fetch_adj_close_monthly_end("^BVSP", start_date, end_date)
    rets_mkt = compute_monthly_returns_from_prices(prices_mkt)
    months_mkt_ret = months_mkt[1:]
    idx_mkt_ret = {months_mkt_ret[i]: i for i in range(len(months_mkt_ret))}
    aligned_mkt = [rets_mkt[idx_mkt_ret[m]] for m in common_months if m in idx_mkt_ret]

    if not aligned_mkt:
        print("Não foi possível alinhar retornos do mercado com os ativos.")
        return {}

    Rm = mean(aligned_mkt)

    for t, rets in aligned_returns.items():
        var_mkt = variance(aligned_mkt)
        beta = (covariance(rets, aligned_mkt) / var_mkt) if var_mkt != 0 else 0.0
        capm = rf + beta * (Rm - rf)
        lista_capm[t] = {"beta": beta, "capm": capm}

    return lista_capm


def fundamental_analysis(tickers):
    resultados = []

    for t in tickers:
        tk = yf.Ticker(t)
        balanco = tk.balance_sheet
        dre = tk.financials

        if balanco.empty or dre.empty:
            print(f"⚠️ Dados contábeis indisponíveis para {t}")
            continue

        for col in balanco.columns:
            ano = col.year
            if ano not in [2021, 2022, 2023, 2024]:
                continue

            ativo_circulante = balanco.get("Total Current Assets", pd.Series()).get(col, 0)
            passivo_circulante = balanco.get("Total Current Liabilities", pd.Series()).get(col, 0)
            estoques = balanco.get("Inventory", pd.Series()).get(col, 0)
            ativo_total = balanco.get("Total Assets", pd.Series()).get(col, 0)
            fornecedores = balanco.get("Accounts Payable", pd.Series()).get(col, 0)
            receita_liquida = dre.get("Total Revenue", pd.Series()).get(col, 0)
            cogs = dre.get("Cost Of Revenue", pd.Series()).get(col, 0)

            liquidez_corrente = ativo_circulante / passivo_circulante if passivo_circulante else None
            liquidez_seca = (ativo_circulante - estoques) / passivo_circulante if passivo_circulante else None
            giro_ativo = receita_liquida / ativo_total if ativo_total else None
            prazo_medio_pagamento = (fornecedores / cogs * 360) if cogs else None

            resultados.append({
                "Empresa": t,
                "Ano": ano,
                "Liquidez Corrente": liquidez_corrente,
                "Liquidez Seca": liquidez_seca,
                "Giro do Ativo": giro_ativo,
                "Prazo Médio de Pagamento (dias)": prazo_medio_pagamento
            })

    df = pd.DataFrame(resultados).sort_values(["Empresa", "Ano"])
    return df



def analysis_for_tickers(tickers, start_date, end_date=None,
                         total_capital=100000.0, weights=None):
    returns_by_ticker = {}
    for t in tickers:
        months, prices = fetch_adj_close_monthly_end(t, start_date, end_date)
        rets = compute_monthly_returns_from_prices(prices)
        returns_by_ticker[t] = (months[1:], rets)

    aligned, common_months = align_return_series(returns_by_ticker)
    per_asset = {}
    for t in tickers:
        _, rets = returns_by_ticker[t]
        mu, sd = mean(rets), stddev(rets)
        cv = coefficient_of_variation(mu, sd)
        per_asset[t] = {
            "mean_return": mu,
            "stddev": sd,
            "cv": cv,
            "cumulative": cumulative_return(rets),
            "n_months": len(rets)
        }

    pairs_corr = {}
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            pairs_corr[(a, b)] = correlation(aligned.get(a, []), aligned.get(b, []))

    w = normalize_weights(weights or {}, tickers)
    port_mean = sum(w[t] * per_asset[t]["mean_return"] for t in tickers)
    port_monthly = compute_portfolio_monthly_returns(aligned, w)
    port_cum = cumulative_return(port_monthly)
    lista_capm = CAPM(aligned, common_months, start_date, end_date)

    def fmt_pct(x): return f"{100 * x:,.3f}%"

    print("\n=== Pesos e Alocação ===")
    for t in tickers:
        print(f"{t}: peso {100 * w[t]:.2f}%   ≈ R$ {total_capital * w[t]:,.2f}")

    print("\n=== Métricas por ativo ===")
    print(f"{'Ticker':<10s} {'n':>6s} {'Média':>12s} {'Desvio':>12s} {'CV':>8s} {'Acumulado':>12s}")
    for t in tickers:
        info = per_asset[t]
        print(
            f"{t:<10s} {info['n_months']:>6d} {fmt_pct(info['mean_return']):>12s} "
            f"{fmt_pct(info['stddev']):>12s} {info['cv']:>8.3f} {fmt_pct(info['cumulative']):>12s}"
        )

    print("\n=== Correlação 2 a 2 ===")
    for (a, b), c in pairs_corr.items():
        print(f"Correlação({a},{b}) = {c: .4f}")

    print("\n=== Carteira ===")
    print(f"Retorno médio mensal: {fmt_pct(port_mean)}")
    if common_months:
        first = f"{common_months[0][0]}-{common_months[0][1]:02d}"
        last = f"{common_months[-1][0]}-{common_months[-1][1]:02d}"
        print(f"Janela (meses comuns): {first} → {last}  |  N = {len(common_months)}")

    print("\nRetornos mensais da CARTEIRA (meses comuns):")
    if not common_months:
        print("Não há meses comuns entre os ativos.")
    else:
        print(f"{'Mês':<10s} {'Retorno':>10s}")
        for (y, m), r in zip(common_months, port_monthly):
            print(f"{y}-{m:02d}   {fmt_pct(r):>10s}")
        print(f"\nRetorno ACUMULADO da carteira: {fmt_pct(port_cum)}")

    print("\n=== CAPM por ativo ===")
    if not lista_capm:
        print("CAPM não pôde ser calculado (mercado não alinhado).")
    else:
        for t, info in lista_capm.items():
            print(f"{t}: Beta = {info['beta']:.4f} | CAPM = {info['capm']*100:.2f}% ao mês")

    return {
        "per_asset": per_asset,
        "pairs_corr": pairs_corr,
        "portfolio_mean": port_mean,
        "portfolio_monthly_returns": port_monthly,
        "portfolio_cumulative_return": port_cum,
        "weights": w,
        "common_months": common_months,
        "aligned_returns": aligned,
        "returns_by_ticker": returns_by_ticker,
        "capm": lista_capm,
    }


if __name__ == "__main__":
    analysis_for_tickers(TICKERS, START_DATE, END_DATE, TOTAL_CAPITAL, WEIGHTS)

    print("\n\n📊 === ANÁLISE FUNDAMENTALISTA (2021–2024) ===")
    df = fundamental_analysis(TICKERS)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("⚠️ Não foi possível obter dados contábeis para as empresas.")
