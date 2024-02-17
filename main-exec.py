from src.pipeline import mdx
from src.trading_data import get_data
import datetime


# Hyperparameters

start='2020-01-01'
end='2022-12-31'
split_date=datetime(2022, 1, 1)
interval_yf='1d'
symbols='BTC-USD'
model_summary=False
verbose=0
BARRIER = [0.2, 0.3, 0.4, 0.5]
UPPER_PROB = [0.4, 0.6, 0.8]
LOWER_PROB = [0.4, 0.6, 0.8]
DAYS_LOOKBACK=30
T_BARRIER =[2, 3, 4, 5, 6, 7, 8 , 9, 10]

hps_results = {
    't_barrier': [],
    'barrier': [],
    'overall_profit': [],
    'n_trades': [],
    'lower_prob': [],
    'upper_prob': []
}

# Get data
df = get_data(
    symbols=[symbols],
    start=start,
    end=end,
    interval=interval_yf
)

for t_barrier in T_BARRIER:
    for barrier in BARRIER:
        for x in LOWER_PROB:
            for z in UPPER_PROB:
                df_2 = df.copy(deep=True)
                overall_profit, n_trades = mdx(
                    df=df_2,
                    barrier_length=t_barrier,
                    barrier_std=barrier,
                    days_lookback=3,
                    split_date=split_date,
                    model_summary=model_summary,
                    verbose=verbose,
                    lower_prob=x,
                    upper_prob=z
                )

                hps_results['t_barrier'].append(t_barrier)
                hps_results['barrier'].append(barrier)
                hps_results['overall_profit'].append(overall_profit)
                hps_results['n_trades'].append(n_trades)
                hps_results['lower_prob'].append(x)
                hps_results['upper_prob'].append(z)
                hp_results = pd.DataFrame(hps_results).sort_values('overall_profit', ascending=False)
                print(hp_results.head())