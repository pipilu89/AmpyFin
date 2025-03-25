import sys

project_name = "AmpyFin_Random_Forest"
# experiment_name = "24year_sp500"
experiment_name = "10year_sp500"
# general parameters
"""
time_delta_mode can be multiplicative, additive, or balanced.
Additive results in less overfitting but could result in underfitting as time goes on
Multiplicative results in more overfitting but less underfitting as time goes on.
Balanced results in a mix of both where time_delta is going to be a fifth of what the current timestamp is
and added to time_Delta so it is less overfitting and less underfitting as time goes on.
time_delta_increment is used for additive purpose
time_delta_multiplicative is used for multiplicative purpose
time_delta_balanced is used for balanced purpose - 0.2 means 0.8 is data influence and 0.2 is current influence.
This is used by both ranking and training clients
"""
time_delta_mode = "additive"
time_delta_increment = 0.01
time_delta_multiplicative = 1.01
time_delta_balanced = 0.2

# helper_files/client_helper.py
"""
stop loss is the percentage of loss you are willing to take before you sell your asset
take profit is the percentage of profit you are willing to take before you sell your asset
these parameters are useful to fine tune your bot
0.03 stop loss means after 3% loss, you will sell your asset
0.05 take profit means after 5% profit, you will sell your asset
"""
stop_loss = 0.03
take_profit = 0.05

# training_client.py parameters
"""
mode is switched between 'train', 'test', live, and 'push'.
'live is the default safe mode. We keep the mode to live to indicate that the bot is in trading / ranking mode.
'train' means running ranking_client.py and getting updated trading_simulator.
There will be an option to:
 - update your database if this is the data you want to insert into the database given better results during test
 - save this model to run testing before you decide to update your database
 - delete this model to start with a new model
'test' means running running your training results on simulator.
'push' means pushing your trained bot to the database. This is only available for the ranking client.
The default for mode is live to protect against accidental training
benchmark asset is what benchmark you want to compare to - typically SPY, QQQ, DOW, or NDAQ.
"""

mode = "train"

benchmark_asset = "SPY"
"""
training parameters - run purely on ranking_client.py
period_start and period_end are the start and end date of the period you want to train
train_tickers are the tickers you want to train on.
if train_tickers is empty, it will train from the current NDAQ holdings
please keep in mind training takes quite a long time.
Our team trained it on a 1m tick, but even on a 1d tick, it takes a really long time
so please understand the time it takes to train.

"""

# Random forest variables
regime_tickers = ["^VIX", "^GSPC"]
prediction_modifier = 1
prediction_threshold = 0.5

# short test period
train_period_start = "2024-12-21"
train_period_end = "2024-12-31"
test_period_start = "2025-01-01"
test_period_end = "2025-02-01"

# train_period_start = "2014-01-01"
# train_period_end = "2024-12-31"
# test_period_start = "2025-01-01"
# test_period_end = "2025-03-01"

train_tickers_5 = ["AAPL", "MSFT", "META", "AMD", "ELV"]
train_tickers1 = ["GOOGL"]
# train_tickers = ["ELV"]
# train_tickers = ['BTC-USD', 'ETH-USD']
nsdaq_tickers = [
    "ADBE",
    "AMAT",
    "CSCO",
    "FAST",
    "MSFT",
    "PAYX",
    "QCOM",
    "AXON",
    "MSTR",
    "PLTR",
    "APP",
    "ARM",
    "LIN",
    "CCEP",
    "DASH",
    "MDB",
    "ROP",
    "TTD",
    "ON",
    "GEHC",
    "BKR",
    "FANG",
    "GFS",
    "WBD",
    "AZN",
    "CEG",
    "ODFL",
    "TEAM",
    "ABNB",
    "FTNT",
    "PANW",
    "ZS",
    "DDOG",
    "CRWD",
    "HON",
    "AEP",
    "NFLX",
    "KDP",
    "PDD",
    "DXCM",
    "ANSS",
    "CDW",
    "CPRT",
    "CSGP",
    "EXC",
    "AMD",
    "LULU",
    "XEL",
    "PEP",
    "ASML",
    "SNPS",
    "TTWO",
    "WDAY",
    "MELI",
    "IDXX",
    "CSX",
    "TMUS",
    "PYPL",
    "KHC",
    "GOOG",
    "NXPI",
    "MAR",
    "CHTR",
    "TSLA",
    "EA",
    "MRVL",
    "REGN",
    "VRSK",
    "META",
    "ADI",
    "MDLZ",
    "TXN",
    "AVGO",
    "BKNG",
    "ADP",
    "ORLY",
    "ROST",
    "MNST",
    "VRTX",
    "ISRG",
    "CDNS",
    "GOOGL",
    "CTSH",
    "ADSK",
    "AMGN",
    "LRCX",
    "CMCSA",
    "GILD",
    "NVDA",
    "AMZN",
    "MCHP",
    "SBUX",
    "MU",
    "INTU",
    "BIIB",
    "AAPL",
    "COST",
    "CTAS",
    "INTC",
    "KLAC",
    "PCAR",
]
sp500_tickers = [
    "MMM",
    "AOS",
    "ABT",
    "ABBV",
    "ACN",
    "ADBE",
    "AMD",
    "AES",
    "AFL",
    "A",
    "APD",
    "ABNB",
    "AKAM",
    "ALB",
    "ARE",
    "ALGN",
    "ALLE",
    "LNT",
    "ALL",
    "GOOGL",
    "GOOG",
    "MO",
    "AMZN",
    "AMCR",
    "AMTM",
    "AEE",
    "AEP",
    "AXP",
    "AIG",
    "AMT",
    "AWK",
    "AMP",
    "AME",
    "AMGN",
    "APH",
    "ADI",
    "ANSS",
    "AON",
    "APA",
    "AAPL",
    "AMAT",
    "APTV",
    "ACGL",
    "ADM",
    "ANET",
    "AJG",
    "AIZ",
    "T",
    "ATO",
    "ADSK",
    "ADP",
    "AZO",
    "AVB",
    "AVY",
    "AXON",
    "BKR",
    "BALL",
    "BAC",
    "BAX",
    "BDX",
    "BRK-B",
    "BBY",
    "TECH",
    "BIIB",
    "BLK",
    "BX",
    "BK",
    "BA",
    "BKNG",
    "BWA",
    "BSX",
    "BMY",
    "AVGO",
    "BR",
    "BRO",
    "BF-B",
    "BLDR",
    "BG",
    "BXP",
    "CHRW",
    "CDNS",
    "CZR",
    "CPT",
    "CPB",
    "COF",
    "CAH",
    "KMX",
    "CCL",
    "CARR",
    # "CTLT",
    "CAT",
    "CBOE",
    "CBRE",
    "CDW",
    "CE",
    "COR",
    "CNC",
    "CNP",
    "CF",
    "CRL",
    "SCHW",
    "CHTR",
    "CVX",
    "CMG",
    "CB",
    "CHD",
    "CI",
    "CINF",
    "CTAS",
    "CSCO",
    "C",
    "CFG",
    "CLX",
    "CME",
    "CMS",
    "KO",
    "CTSH",
    "CL",
    "CMCSA",
    "CAG",
    "COP",
    "ED",
    "STZ",
    "CEG",
    "COO",
    "CPRT",
    "GLW",
    "CPAY",
    "CTVA",
    "CSGP",
    "COST",
    "CTRA",
    "CRWD",
    "CCI",
    "CSX",
    "CMI",
    "CVS",
    "DHR",
    "DRI",
    "DVA",
    "DAY",
    "DECK",
    "DE",
    "DELL",
    "DAL",
    "DVN",
    "DXCM",
    "FANG",
    "DLR",
    "DFS",
    "DG",
    "DLTR",
    "D",
    "DPZ",
    "DOV",
    "DOW",
    "DHI",
    "DTE",
    "DUK",
    "DD",
    "EMN",
    "ETN",
    "EBAY",
    "ECL",
    "EIX",
    "EW",
    "EA",
    "ELV",
    "EMR",
    "ENPH",
    "ETR",
    "EOG",
    "EPAM",
    "EQT",
    "EFX",
    "EQIX",
    "EQR",
    "ERIE",
    "ESS",
    "EL",
    "EG",
    "EVRG",
    "ES",
    "EXC",
    "EXPE",
    "EXPD",
    "EXR",
    "XOM",
    "FFIV",
    "FDS",
    "FICO",
    "FAST",
    "FRT",
    "FDX",
    "FIS",
    "FITB",
    "FSLR",
    "FE",
    "FI",
    "FMC",
    "F",
    "FTNT",
    "FTV",
    "FOXA",
    "FOX",
    "BEN",
    "FCX",
    "GRMN",
    "IT",
    "GE",
    "GEHC",
    "GEV",
    "GEN",
    "GNRC",
    "GD",
    "GIS",
    "GM",
    "GPC",
    "GILD",
    "GPN",
    "GL",
    "GDDY",
    "GS",
    "HAL",
    "HIG",
    "HAS",
    "HCA",
    "DOC",
    "HSIC",
    "HSY",
    "HES",
    "HPE",
    "HLT",
    "HOLX",
    "HD",
    "HON",
    "HRL",
    "HST",
    "HWM",
    "HPQ",
    # "HUBB",
    "HUM",
    "HBAN",
    "HII",
    "IBM",
    "IEX",
    "IDXX",
    "ITW",
    "INCY",
    "IR",
    "PODD",
    "INTC",
    "ICE",
    "IFF",
    "IP",
    "IPG",
    "INTU",
    "ISRG",
    "IVZ",
    "INVH",
    "IQV",
    "IRM",
    "JBHT",
    "JBL",
    "JKHY",
    "J",
    "JNJ",
    "JCI",
    "JPM",
    "JNPR",
    "K",
    "KVUE",
    "KDP",
    "KEY",
    "KEYS",
    "KMB",
    "KIM",
    "KMI",
    "KKR",
    "KLAC",
    "KHC",
    "KR",
    "LHX",
    "LH",
    "LRCX",
    "LW",
    "LVS",
    "LDOS",
    "LEN",
    "LLY",
    "LIN",
    "LYV",
    "LKQ",
    "LMT",
    "L",
    "LOW",
    "LULU",
    "LYB",
    "MTB",
    "MPC",
    "MKTX",
    "MAR",
    "MMC",
    "MLM",
    "MAS",
    "MA",
    "MTCH",
    "MKC",
    "MCD",
    "MCK",
    "MDT",
    "MRK",
    "META",
    "MET",
    "MTD",
    "MGM",
    "MCHP",
    "MU",
    "MSFT",
    "MAA",
    "MRNA",
    "MHK",
    "MOH",
    "TAP",
    "MDLZ",
    "MPWR",
    "MNST",
    "MCO",
    "MS",
    "MOS",
    "MSI",
    "MSCI",
    "NDAQ",
    "NTAP",
    "NFLX",
    "NEM",
    "NWSA",
    "NWS",
    "NEE",
    "NKE",
    "NI",
    "NDSN",
    "NSC",
    "NTRS",
    "NOC",
    "NCLH",
    "NRG",
    "NUE",
    "NVDA",
    "NVR",
    "NXPI",
    "ORLY",
    "OXY",
    "ODFL",
    "OMC",
    "ON",
    "OKE",
    "ORCL",
    "OTIS",
    "PCAR",
    "PKG",
    "PLTR",
    "PANW",
    "PARA",
    "PH",
    "PAYX",
    "PAYC",
    "PYPL",
    "PNR",
    "PEP",
    "PFE",
    "PCG",
    "PM",
    "PSX",
    "PNW",
    "PNC",
    "POOL",
    "PPG",
    "PPL",
    "PFG",
    "PG",
    "PGR",
    "PLD",
    "PRU",
    "PEG",
    "PTC",
    "PSA",
    "PHM",
    "QRVO",
    "PWR",
    "QCOM",
    "DGX",
    "RL",
    "RJF",
    "RTX",
    "O",
    "REG",
    "REGN",
    "RF",
    "RSG",
    "RMD",
    "RVTY",
    "ROK",
    "ROL",
    "ROP",
    "ROST",
    "RCL",
    "SPGI",
    "CRM",
    "SBAC",
    "SLB",
    "STX",
    "SRE",
    "NOW",
    "SHW",
    "SPG",
    "SWKS",
    "SJM",
    "SW",
    "SNA",
    "SOLV",
    "SO",
    "LUV",
    "SWK",
    "SBUX",
    "STT",
    "STLD",
    "STE",
    "SYK",
    "SMCI",
    "SYF",
    "SNPS",
    "SYY",
    "TMUS",
    "TROW",
    "TTWO",
    "TPR",
    "TRGP",
    "TGT",
    "TEL",
    "TDY",
    "TFX",
    "TER",
    "TSLA",
    "TXN",
    "TPL",
    "TXT",
    "TMO",
    "TJX",
    "TSCO",
    "TT",
    "TDG",
    "TRV",
    "TRMB",
    "TFC",
    "TYL",
    "TSN",
    "USB",
    "UBER",
    "UDR",
    "ULTA",
    "UNP",
    "UAL",
    "UPS",
    "URI",
    "UNH",
    "UHS",
    "VLO",
    "VTR",
    "VLTO",
    "VRSN",
    "VRSK",
    "VZ",
    "VRTX",
    "VTRS",
    "VICI",
    "V",
    "VST",
    "VMC",
    "WRB",
    "GWW",
    "WAB",
    "WBA",
    "WMT",
    "DIS",
    "WBD",
    "WM",
    "WAT",
    "WEC",
    "WFC",
    "WELL",
    "WST",
    "WDC",
    "WY",
    "WMB",
    "WTW",
    "WYNN",
    "XEL",
    "XYL",
    "YUM",
    "ZBRA",
    "ZBH",
    "ZTS",
]

train_tickers = train_tickers_5
"""
train_time_delta_mode can be multiplicative, additive, or balanced.
Additive results in less overfitting but could result in underfitting as time goes on
Multiplicative results in more overfitting but less underfitting as time goes on.
Balanced results in a mix of both where time_delta is going to be a fifth of what the current timestamp is
and added to time_Delta so it is less overfitting and less underfitting as time goes on.
train_time_delta_increment is used for additive purpose.
train_time_delta_multiplicative is used for multiplicative purpose
train_time_delta_balanced is used for balanced purpose - 0.1 means 0.9 is data influence and 0.1 is current influence
train time delta is the starting time delta for the training client
"""
train_time_delta = sys.float_info.min
train_time_delta_mode = "balanced"
train_time_delta_increment = 0.01
train_time_delta_multiplicative = 1.01
train_time_delta_balanced = 0.2

"""
train suggestion_heap_limit - at what threshold of buy_weight limit should the ticker be considered for suggestion
"""
train_suggestion_heap_limit = 600000

"""
train_start_cash - the starting cash for the training client
"""
train_start_cash = 50000.00

"""
train_trade_liquidity_limit is the amount of money you are telling the bot to reserve during trading.
All bots start with a default of 50000 as liquidity with limit as specified here. This is for the training client.
"""
train_trade_liquidity_limit = 15000.00

"""
train_trade_asset_limit to portfolio is how much asset
you are allowed to hold in comparison to portfolio value for the training client during trading
The lower this number, the more diversification you will have in your portfolio. The higher the number,
the less diversification you will have but it will be buying more selective assets.
"""
train_trade_asset_limit = 0.1
train_trade_strategy_limit = 0.2

"""
train_rank_liquidity_limit is the amount of money you are telling the bot to reserve during ranking.
All bots start with a default of 50000 as liquidity with limit as specified here. This is for the training client.
"""
train_rank_liquidity_limit = 15000

"""
train_rank_asset_limit to portfolio is how much asset
you are allowed to hold in comparison to portfolio value for the training client during ranking
The lower this number, the more diversification you will have in your portfolio. The higher the number,
the less diversification you will have but it will be buying more selective assets.
"""
train_rank_asset_limit = 0.3

"""
train_profit_price_change_ratio_(d1 - d2) is at what price ratio you should reward each strategy
train_profit_profit_time_(d1 - d2) is how much reward you should give to the strategy.
For example profit_price_change_ratio_d1 = 1.01 and profit_profit_time_d1 = 1.1 means that if
the price of the asset goes up but less than by 1% in the trade during sell,
you should reward the strategy by multiple of time_delta * 1.1
train_profit_price_delta_else is the reward you should give to the strategy is it exceeds profit_price_change_ratio_d2
"""
train_profit_price_change_ratio_d1 = 1.05
train_profit_profit_time_d1 = 1
train_profit_price_change_ratio_d2 = 1.1
train_profit_profit_time_d2 = 1.5
train_profit_profit_time_else = 1.2

"""
loss_price_change_ratio_(d1 - d2) defines at what price ratio you should penalize each strategy.
loss_profit_time_(d1 - d2) determines how much penalty you should give to the strategy.
For example, loss_price_change_ratio_d1 = 0.99 and loss_profit_time_d1 = 1 means that if
the price of the asset goes down but by less than 1% in the trade during sell,
you should penalize the strategy by a multiple of time_delta * 1.
loss_price_delta_else is the penalty you should apply if the loss exceeds loss_price_change_ratio_d2.
"""
train_loss_price_change_ratio_d1 = 0.975
train_loss_profit_time_d1 = 1
train_loss_price_change_ratio_d2 = 0.95
train_loss_profit_time_d2 = 1.5
train_loss_profit_time_else = 2

"""
train_stop_loss - the percentage of loss you are willing to take before you sell your asset
train_take_profit - the percentage of profit you are willing to take before you sell your asset
"""
train_stop_loss = 0.03
train_take_profit = 0.05

# ranking_client.py parameters

"""
rank_liquidity_limit is the amount of money you are telling the bot to reserve during ranking.
All bots start with a default of 50000 as liquidity with limit as specified here. This is for the ranking client.
"""
rank_liquidity_limit = 15000

"""
rank_asset_limit to portfolio is how much asset you are allowed to hold in comparison to portfolio value for the ranking client
The lower this number, the more diversification you will have in your portfolio. The higher the number,
the less diversification you will have but it will be buying more selective assets.
"""
rank_asset_limit = 0.1

"""
profit_price_change_ratio_(d1 - d2) is at what price ratio you should reward each strategy
profit_profit_time_(d1 - d2) is how much reward you should give to the strategy.
For example profit_price_change_ratio_d1 = 1.01 and profit_profit_time_d1 = 1.1 means that if
the price of the asset goes up but less than by 1% in the trade during sell,
you should reward the strategy by multiple of time_delta * 1.1
profit_price_delta_else is the reward you should give to the strategy is it exceeds profit_price_change_ratio_d2
"""
profit_price_change_ratio_d1 = 1.05
profit_profit_time_d1 = 1
profit_price_change_ratio_d2 = 1.1
profit_profit_time_d2 = 1.5
profit_profit_time_else = 1.2

"""
loss_price_change_ratio_(d1 - d2) defines at what price ratio you should penalize each strategy.
loss_profit_time_(d1 - d2) determines how much penalty you should give to the strategy.
For example, loss_price_change_ratio_d1 = 0.99 and loss_profit_time_d1 = 1 means that if
the price of the asset goes down but by less than 1% in the trade during sell,
you should penalize the strategy by a multiple of time_delta * 1.
loss_price_delta_else is the penalty you should apply if the loss exceeds loss_price_change_ratio_d2.
"""
loss_price_change_ratio_d1 = 0.975
loss_profit_time_d1 = 1
loss_price_change_ratio_d2 = 0.95
loss_profit_time_d2 = 1.5
loss_profit_time_else = 2

# trading_client.py parameters
"""
trade_liquidity_limit is the amount of money you are telling the bot to reserve during ranking.
All bots start with a default of 50000. This is for the trading client. Please try not to change this.
If you do, the suggestion for bottom limit is 20% of the portfolio value.
"""
trade_liquidity_limit = 15000

"""
trade_asset_limit to portfolio is how much asset you are allowed to hold in comparison to portfolio value for the trading client
The lower this number, the more diversification you will have in your portfolio. The higher the number,
the less diversification you will have but it will be buying more selective assets.
Thsi will also be reflected in Ta-Lib for suggestion and could also affect ranking as well in terms of asset_limit
"""
trade_asset_limit = 0.1

"""
suggestion heap is used in case of when the trading system becomes overpragmatic.
This is at what buy_weight limit should the ticker be considered for suggestion
to buy if the system is pragmatic on all other tickers.
"""
suggestion_heap_limit = 600000

"""
when we train, it will be running ranking_client.py

when we backtest, it will be running training_client.pt and ranking_client.py simultaenously
"""
