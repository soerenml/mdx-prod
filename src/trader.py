def execute_strategy(
    data,
    upper_prob,
    lower_prob
):
    portfolio = 0
    stocks = 0
    n_trades = 0
    entry_price = 0
    total_profit = 0
    overall_profit = 0
    trading_cost: float = 0.01
    total_trading_cost: float = 0
    holding_stock = False

    data['profit'] = None
    data['total_profit'] = None
    for index, row in data.iterrows():
        close_price = row.get('close', None)
        strategy = row.get('strategy', None)
        prob_strat = row.get('prob_strat', None)
        date = row.get('date', None)
        if strategy == 'l_buy' \
          and prob_strat >= upper_prob \
          and holding_stock == False:
            # Buy the stock only if not holding any
            entry_price = close_price
            holding_stock = True
            stocks += 1
            n_trades += 1
            portfolio += entry_price
            #print(
            #    f"""Buying at index {index},
            #    entry price: {entry_price},
            #    portfolio: {portfolio},
            #    number of stocks: {stocks},
            #    date: {date}"""
            #)
        elif strategy == 'l_sell' and holding_stock == True \
          and close_price is not None \
          and prob_strat >= lower_prob:
            # Sell the stock only if holding any
            exit_price = close_price
            profit = ((exit_price/portfolio)-1)*100
            total_profit += profit
            portfolio += profit
            overall_profit += profit
            n_trades += 1
            total_trading_cost += (entry_price + exit_price) * trading_cost
            #print(
            #    f"""Selling at index {index},
            #    entry price: {entry_price},
            #    exit price: {exit_price},
            #    profit: {profit},
            #    overall_profit: {overall_profit},
            #    date: {date}"""
            #)
            holding_stock = False
            stocks = 0
            portfolio = 0
    #print(f"Overall profit: {overall_profit}, trading costs: {total_trading_cost}")
    overall_profit += total_profit
    return overall_profit, n_trades