from matplotlib import table
import pandas as pd
import yfinance as yf
import sqlite3
import os
import sys
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    DateOffset,
    EasterMonday,
    GoodFriday,
    Holiday,
    MO,
    next_monday,
    next_monday_or_tuesday,
)  # https://www.tobiolabode.com/blog/2019/1/1/pandas-for-uk-hoildays
import logging

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from control import train_tickers, regime_tickers

# Set up logging
logs_dir = "logs"
# Create the directory if it doesn't exist
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

file_handler = logging.FileHandler(os.path.join(logs_dir, "price_data.log"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# database connection
# database_name = "price_data.db"
last_dl_date_table_name = "last_dl_date"
db_path = r"c:\Users\pi\code\python-t212\price_data.db"
con = sqlite3.connect(db_path)
# BASE_PATH = os.getenv("BASE_PATH")
# last_dl_date_table_name = os.getenv("last_dl_date_table_name")
# database_name = os.getenv("database_name")
# con = sqlite3.connect(database_name)


# index vs primary key for sql to avoid duplicates?

# https://pypi.org/project/yfinance/  To download price history into one table:
# data = yf.download("SPY AAPL", period="1mo")
# logger.info(data)
# https://stackoverflow.com/questions/63107594/how-to-deal-with-multi-level-column-names-downloaded-with-yfinance/63107801#63107801


def df_to_sql(df, table_name):
    with con:
        # with sqlite3.connect(database_name) as con:
        df.to_sql(table_name, con, if_exists="append")
        # df.to_sql(table_name, con)


def sql_to_df(table_name):
    with con:
        # with sqlite3.connect(database_name) as con:
        df = pd.read_sql_query(
            "SELECT * FROM `{tab}`".format(tab=table_name),
            con,
            index_col="Date",
        )
        # logger.info(df.shape)
        # logger.info(df)
    return df


def sql_to_df_with_date_range(table_name, begin_date, end_date):
    with con:
        # with sqlite3.connect(database_name) as con:
        df = pd.read_sql_query(
            "select * from `{tab}` WHERE `Date` >= '{begin}' AND `Date` <= '{end}'".format(
                tab=table_name, begin=begin_date, end=end_date
            ),
            con,
            index_col="Date",
        )
    return df


def sql_date_to_close_price(table_name, begin_date):
    with con:
        # with sqlite3.connect(database_name) as con:
        # df = pd.read_sql_query("select * from `{tab}` WHERE `Date` == '{begin}'".format(tab=table_name, begin=begin_date), con, index_col='Date')
        cur = con.cursor()
        sqlite_select_query = (
            "select Close from `{tab}` WHERE `Date` == '{begin}'".format(
                tab=table_name, begin=begin_date
            )
        )
        cur.execute(sqlite_select_query)
        record = cur.fetchone()
        # logger.info(f'dl date {record=}')
        if record == None:
            return None
        # dl_date = record[1]
        dl_date = record[0]
        # logger.info(f'pie_name: {record[0]}, {dl_date=}')
        # con.commit()
        return dl_date


def sql_to_df_with_end_date(table_name, end_date):
    with con:
        # with sqlite3.connect(database_name) as con:
        df = pd.read_sql_query(
            "select * from `{tab}` WHERE `Date` <= '{end}'".format(
                tab=table_name, end=end_date
            ),
            con,
            index_col="Date",
        )
    return df


def get_table_names():
    with con:
        # with sqlite3.connect(database_name) as con:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master")
        table_names = cur.fetchall()
        # logger.info(f'{table_names=}')
        return table_names


def create_table_schema(table_name):
    try:
        # logger.info(f'try to create table: {table_name}...')
        with con:
            # with sqlite3.connect(database_name) as con:
            cur = con.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS `{tab}`
        (Date TEXT PRIMARY KEY     NOT NULL,
        Open  REAL,
        High  REAL,
        Low REAL,
        Close REAL,
        Volume  REAL) WITHOUT ROWID;""".format(
                    tab=table_name
                )
            )
            # print ("Table created successfully")
            con.commit()
            return table_name
    except sqlite3.Error as e:
        logger.info(e)
        return None


def create_dl_date_table_schema(table_name):
    try:
        # logger.info(f'try to create table: {table_name}...')
        with con:
            # with sqlite3.connect(database_name) as con:
            cur = con.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS `{tab}`
        (pie_name TEXT PRIMARY KEY     NOT NULL,
        dl_date  REAL) WITHOUT ROWID;""".format(
                    tab=table_name
                )
            )
            # print ("Table created successfully")
            con.commit()
            return table_name
    except sqlite3.Error as e:
        logger.info(e)
        return None


def print_table(table):
    with con:
        # with sqlite3.connect(database_name) as con:
        logger.info(f"print data from table: {table}")
        cur = con.cursor()
        cur.execute("SELECT * FROM `{tab}`".format(tab=table))
        rows = cur.fetchall()
        for row in rows:
            logger.info(row)
        if len(rows) == 0:
            logger.info("table empty")


def table_schema(table_name):
    with con:
        # with sqlite3.connect(database_name) as con:
        cur = con.cursor()
        cur.execute("""SELECT sql FROM sqlite_schema WHERE name = ?;""", (table_name,))
        table_schema = cur.fetchall()
        logger.info(f"{table_schema=}")


def drop_table(table_name):
    with con:
        # with sqlite3.connect(database_name) as con:
        cur = con.cursor()
        cur.execute("""DROP TABLE IF EXISTS `{tab}`""".format(tab=table_name))
        logger.info("table dropped successfully")


def upsert(table_name, data):
    try:
        # logger.info(f'try upsert: {table_name}...')
        with con:
            # with sqlite3.connect(database_name) as con:
            cur = con.cursor()
            cur.executemany(
                """INSERT INTO `{tab}`
                  VALUES(?, ?, ?, ?, ?, ?)
                  ON CONFLICT
                  DO 
                  UPDATE SET 
                      Open=excluded.Open,
                      High=excluded.High,
                      Low=excluded.Low,
                      Close=excluded.Close,
                      Volume=excluded.Volume;""".format(
                    tab=table_name
                ),
                data,
            )
            con.commit()
            # return table_name
    except sqlite3.Error as e:
        logger.error(e)
        return None


def upsert_date_today(table_name, data):
    try:
        logger.info(f"try upsert: {table_name}, {data=}...")
        with con:
            # with sqlite3.connect(database_name) as con:
            cur = con.cursor()
            cur.executemany(
                """INSERT INTO `{tab}`
                  VALUES(?, ?)
                  ON CONFLICT
                  DO 
                  UPDATE SET 
                      dl_date=excluded.dl_date;""".format(
                    tab=table_name
                ),
                data,
            )
            con.commit()
            # return table_name
    except sqlite3.Error as e:
        logger.info(e)
        return None


def readSingleRow(table_name, pie_name):
    try:
        # logger.info(f'reading dl date for {pie_name} from {table_name} sql table...')
        with con:
            # with sqlite3.connect(database_name) as con:
            cur = con.cursor()
            sqlite_select_query = (
                """SELECT * from `{table}` where pie_name = ?""".format(
                    table=table_name
                )
            )
            cur.execute(sqlite_select_query, (pie_name,))
            record = cur.fetchone()
            logger.info(f"dl date {record=}")
            if record == None:
                return ""
            dl_date = record[1]
            # logger.info(f'pie_name: {record[0]}, {dl_date=}')
            # con.commit()
            return dl_date
    except sqlite3.Error as e:
        logger.info(e)
        return None


def convert_df_to_sql_values(df):
    df.index = df.index.astype(str)
    sql_values = list(df.itertuples(index=True, name=None))
    # sql_values = df.to_records(index=True).tolist()
    # sql_values = [tuple(x) for x in df.to_records(index=True)]
    # sql_values = list(df.to_records())
    # logger.info(f'{sql_values=}')
    return sql_values


def download_and_store(df_tickers, options, pie_name):
    today = pd.to_datetime("today").date().strftime("%Y-%m-%d")

    def download_data_from_yf2(df_tickers, yf_period, pie_name):
        logger.info(f"\n---DL DATA---{pie_name}---{today}")
        if readSingleRow(last_dl_date_table_name, pie_name) == today:
            logger.info("data already downloaded today")
            return

        logger.info("download data...")
        for index, row in tqdm(df_tickers.iterrows()):
            ticker = row["yahoo_ticker"]
            create_table_schema(ticker)
            try:
                # df = yf.download(ticker, period = yf_period, interval = '1d', auto_adjust=True, progress=False)
                df = yf.download(
                    ticker,
                    period=yf_period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    repair=True,
                    rounding=True,
                )
            except Exception as e:
                logger.error("yf error")
                logger.error(e)
            df = df.round(2)
            df = df.drop(
                ["Repaired?"], axis=1, errors="ignore"
            )  # if yf repairs data it adds this col.
            sql_values = convert_df_to_sql_values(df)
            upsert(ticker, sql_values)

        # update db with todays dl date
        data = [(pie_name, today)]
        # store_dl_date(last_dl_date_table_name, data)
        create_dl_date_table_schema(last_dl_date_table_name)
        upsert_date_today(last_dl_date_table_name, data)

    def download_data_from_yf(df_tickers, yf_period, pie_name):
        logger.info(f"\n---DL DATA---{pie_name}---{today}")
        if readSingleRow(last_dl_date_table_name, pie_name) == today:
            logger.info("data already downloaded today")
            return

        logger.info("download data...")
        # ticker_list = list(df_tickers["yahoo_ticker"])
        ticker_list = df_tickers  # modification for ampy
        try:
            df = yf.download(
                ticker_list,
                group_by="Ticker",
                period=yf_period,
                interval="1d",
                auto_adjust=True,
                repair=True,
                rounding=True,
            )
        except Exception as e:
            logger.error("yf error")
            logger.error(e)

        # stack multi-level column index
        df = (
            df.stack(level=0, future_stack=True)
            .rename_axis(["Date", "Ticker"])
            .reset_index(level=1)
        )

        for ticker in ticker_list:
            # drop_table(ticker)
            create_table_schema(ticker)
            df_single_ticker = df[["Open", "High", "Low", "Close", "Volume"]].loc[
                df["Ticker"] == ticker
            ]
            df_single_ticker = df_single_ticker.dropna()
            # logger.info(f"{df_single_ticker = }")
            sql_values = convert_df_to_sql_values(df_single_ticker)
            # logger.info(f"{sql_values = }")
            upsert(ticker, sql_values)

        # update db with todays dl date
        data = [(pie_name, today)]
        # store_dl_date(last_dl_date_table_name, data)
        create_dl_date_table_schema(last_dl_date_table_name)
        upsert_date_today(last_dl_date_table_name, data)

    # yf_period_offset = 20 # avoid error from occansional missing dates in yf.
    if "backtest_offset_yf_period" in options:
        # yf_period = f"{options['backtest_offset_yf_period']+yf_period_offset}d"
        yf_period = "max"
    else:
        yf_period = "1y"
    # elif options['regression_period'] > options['ma_period']:
    #   yf_period = f"{options['regression_period']+yf_period_offset}d"
    # else:
    #   yf_period = f"{options['ma_period']+yf_period_offset}d"
    download_data_from_yf(df_tickers, yf_period, pie_name)  # tickers

    if options["trend_symbol"] != "":
        df_trend_symbol = pd.DataFrame(
            {"yahoo_ticker": options["trend_symbol"]}, index=[0]
        )
        if "backtest_offset_yf_period" in options:
            # yf_trend_period = f"{options['backtest_offset_yf_period']+yf_period_offset}d"
            yf_trend_period = "max"
        else:
            # yf_trend_period = f"{options['trend_period']+yf_period_offset}d"
            yf_trend_period = "1y"
        download_data_from_yf(
            df_trend_symbol, yf_trend_period, options["trend_symbol"]
        )  # trend

    if "benchmark" in options:
        yf_period = "max"
        df_benchmark = pd.DataFrame({"yahoo_ticker": options["benchmark"]}, index=[0])
        download_data_from_yf(df_benchmark, yf_period, options["benchmark"])


"""
DATA CHECKS
"""


def check_missing_dates(table_name, periods_int, exchange="NYSE"):
    # dates which are not in the sequence are returned
    # custom business day range: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#custom-business-days
    df = sql_to_df(table_name)
    # logger.debug(f"{len(df) = }, {df.empty = }")
    df.index = pd.to_datetime(df.index)
    today = pd.to_datetime("today").date()
    # logger.info(f'{today=}')
    # date_range = pd.date_range(end=today, periods=periods_int, freq='B')
    if df.empty:
        earliest_date = None
    else:
        earliest_date = df.index[0]
    # logger.debug(f"{earliest_date = }")

    if exchange == "NYSE":
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        freq = us_bd
    elif exchange == "LSE":

        class Hoildays_England_and_Wales(AbstractHolidayCalendar):
            rules = [
                Holiday("New Years Day", month=1, day=1, observance=next_monday),
                GoodFriday,
                EasterMonday,
                Holiday(
                    "Early May Bank Holiday",
                    month=5,
                    day=1,
                    offset=DateOffset(weekday=MO(1)),
                ),
                Holiday(
                    "Spring Bank Holiday",
                    month=5,
                    day=31,
                    offset=DateOffset(weekday=MO(-1)),
                ),
                Holiday(
                    "Summer Bank Holiday",
                    month=8,
                    day=31,
                    offset=DateOffset(weekday=MO(-1)),
                ),
                Holiday("Christmas Day", month=12, day=25, observance=next_monday),
                Holiday(
                    "Boxing Day", month=12, day=26, observance=next_monday_or_tuesday
                ),
            ]

        uk_bd = CustomBusinessDay(calendar=Hoildays_England_and_Wales())
        freq = uk_bd
    else:
        logger.debug("exchange not recognised, default to NYSE")
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        freq = us_bd

    date_range = pd.date_range(end=today, periods=periods_int, freq=freq)

    # logger.debug(f"{table_name = }, {exchange = }")
    # logger.debug(f"{table_name = }, {date_range = }")
    number_of_missing_dates = len(date_range.difference(df.index))
    missing_date_list = date_range.difference(df.index).strftime("%Y-%m-%d").tolist()
    # logger.info(date_range.difference(df.index).strftime("%Y-%m-%d").tolist())
    # if number_of_missing_dates > 0:
    #   print(f'{table_name} {df.shape=} {number_of_missing_dates=}, {missing_date_list = }')
    # logger.info(date_range.difference(df.index))
    return number_of_missing_dates, missing_date_list, earliest_date


def check_missing_dates_start_end(table_name, start_date, end_date, exchange="us"):
    # dates which are not in the sequence are returned
    # custom business day range: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#custom-business-days
    df = sql_to_df(table_name)
    df.index = pd.to_datetime(df.index)
    today = pd.to_datetime("today").date()
    # logger.info(f'{today=}')
    # date_range = pd.date_range(end=today, periods=periods_int, freq='B')

    if exchange == "us":
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        # date_range = pd.date_range(end=today, periods=periods_int, freq=us_bd)
        date_range = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    elif exchange == "uk":

        class Hoildays_England_and_Wales(AbstractHolidayCalendar):
            rules = [
                Holiday("New Years Day", month=1, day=1, observance=next_monday),
                GoodFriday,
                Holiday(
                    "Early May Bank Holiday",
                    month=5,
                    day=1,
                    offset=DateOffset(weekday=MO(1)),
                ),
                Holiday(
                    "Spring Bank Holiday",
                    month=5,
                    day=31,
                    offset=DateOffset(weekday=MO(-1)),
                ),
                Holiday(
                    "Summer Bank Holiday",
                    month=8,
                    day=31,
                    offset=DateOffset(weekday=MO(-1)),
                ),
                Holiday("Christmas Day", month=12, day=25, observance=next_monday),
                Holiday(
                    "Boxing Day", month=12, day=26, observance=next_monday_or_tuesday
                ),
            ]

        uk_bd = CustomBusinessDay(calendar=Hoildays_England_and_Wales())
        # date_range = pd.date_range(end=today, periods=periods_int, freq=uk_bd)
        date_range = pd.date_range(start=start_date, end=end_date, freq=uk_bd)

    number_of_missing_dates = len(date_range.difference(df.index))
    if number_of_missing_dates > 0:
        logger.info(f"{table_name} {df.shape=} {number_of_missing_dates=}")
        # logger.info(date_range.difference(df.index))
        # logger.info(date_range.difference(df.index).strftime("%Y-%m-%d").tolist())
    return number_of_missing_dates


def check_data(df_tickers, period):
    """
    period: how many days to look back from today. ie check recent price data exists.

    recent data most important. ie has it dled last weeks/ month data?  # data completeness in last 7/30 days  # periods_recent = 7
    """
    logger.info("\n---CHECK DATA--checking recent data...")

    df_tickers = df_tickers.copy()
    (
        df_tickers.loc[:, "#_recent_missing_dates"],
        df_tickers.loc[:, "recent_list_of_missing_dates"],
        df_tickers.loc[:, "earliest_date"],
    ) = zip(
        *df_tickers.apply(
            lambda row: check_missing_dates(
                row["yahoo_ticker"], period, row["exchange"]
            ),
            axis=1,
        )
    )

    # df_tickers["#_recent_missing_dates"], df_tickers["recent_list_of_missing_dates"], df_tickers["earliest_date"] = zip(*df_tickers.apply(lambda row: check_missing_dates(row['yahoo_ticker'], period, row['exchange']), axis=1))

    logger.debug(
        f"{df_tickers[['exchange', '#_recent_missing_dates', 'recent_list_of_missing_dates', 'earliest_date']]}"
    )
    num_tickers_complete_data = len(
        df_tickers[df_tickers["#_recent_missing_dates"] == 0]
    )
    num_tickers_incomplete_data = len(
        df_tickers[df_tickers["#_recent_missing_dates"] != 0]
    )
    pct_with_complete_data = num_tickers_complete_data / len(df_tickers)
    logger.info(
        f"{num_tickers_complete_data = }, {num_tickers_incomplete_data = }, {pct_with_complete_data = }"
    )
    return pct_with_complete_data


# When does historical data start? What date can we backtest from?

# period = '7d'
# periods_int = 7
# periods_int = int(''.join(filter(str.isdigit, period)))
# df_tickers = pd.read_excel(f'{os.path.join(BASE_PATH, "gem_t212_tickers.xlsx")}', index_col=0)
# download_and_store(df_tickers, period)


# for index, row in df_tickers.iterrows():
#     ticker = row['yahoo_ticker']
#     # logger.info(ticker)
#     number_of_missing_dates = check_missing_dates(ticker, periods_int, 'uk')


def check_data_if_missing_dl(df_tickers, period):
    periods_int = int("".join(filter(str.isdigit, period)))
    # check integrity of data.
    for index, row in df_tickers.iterrows():
        ticker = row["yahoo_ticker"]

        # check table exists (but some tickers arent dled eg LSE)

        # logger.info(ticker)
        number_of_missing_dates = check_missing_dates(ticker, periods_int, "uk")
        if number_of_missing_dates > 0:
            download_and_store(df_tickers, period)


# check_data_if_missing_dl(df_tickers, period)


if __name__ == "__main__":
    df_tickers = train_tickers + regime_tickers
    options = {
        # "asset_class": "us_equities",
        # "period": "105d",  # yf download period
        # "regression_period": 90,
        # "ma_period": 100,
        # "gap_threshold": 1,
        # "volume_threshold": 5000,
        "trend_symbol": "",
        # "trend_period": 200,
        # "equityQty": 25,
        # "rebalance_freq": "weekly",
        # "benchmark": "^STOXX",
        "backtest_offset_yf_period": "max",
    }
    pie_name = "ampy"
    download_and_store(df_tickers, options, pie_name)

    # create ticker price history from db.
    table_name = "AAPL"
    begin_date = "2021-01-04"
    end_date = "2021-01-29"

    # ticker_price_history = {}
    # for ticker in df_tickers:
    #     ticker_price_history[ticker] = sql_to_df_with_date_range(
    #         ticker, begin_date, end_date
    #     )

    # print(ticker_price_history)
