import sqlite3
import logging
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper_files.client_helper import setup_logging


def summarize_strategy_decisions(db_path):
    """
    Summarizes the number of 'buy', 'hold', and 'sell' decisions in each table of the strategy_decisions.db.

    Args:
        db_path (str): The path to the strategy_decisions.db file.

    Returns:
        pd.DataFrame: A DataFrame containing the summary data.
    """
    logger = setup_logging("logs", "summary.log", level=logging.INFO)
    summary_data = []  # List to store summary data for each table
    con = None  # Initialize con to None
    try:
        con = sqlite3.connect(db_path)
        cursor = con.cursor()

        # Get a list of all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        # Remove 'summary' from the list if it exists
        if "summary" in tables:
            tables.remove("summary")

        for table in tables:
            logger.info(f"Summary for table: {table}")
            try:
                # Query to count 'buy', 'hold', and 'sell' decisions
                # SQLite does not support UNPIVOT. We need to use a different approach.
                # This approach assumes that all columns except 'Date' contain the decision values.

                cursor.execute(
                    f"SELECT name FROM PRAGMA_TABLE_INFO('{table}') WHERE name <> 'Date'"
                )
                ticker_columns = [row[0] for row in cursor.fetchall()]

                if not ticker_columns:
                    logger.warning(f"No ticker columns found in table: {table}")
                    continue

                buy_count = 0
                hold_count = 0
                sell_count = 0

                for ticker in ticker_columns:
                    query = f"SELECT `{ticker}` FROM `{table}`"
                    df = pd.read_sql_query(query, con)

                    if not df.empty:
                        buy_count += df[ticker].tolist().count("Buy")
                        hold_count += df[ticker].tolist().count("Hold")
                        sell_count += df[ticker].tolist().count("Sell")

                logger.info(f"Buy: {buy_count}")
                logger.info(f"Hold: {hold_count}")
                logger.info(f"Sell: {sell_count}")

                summary_data.append(
                    {
                        "Table": table,
                        "Hold": hold_count,
                        "Buy": buy_count,
                        "Sell": sell_count,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing table {table}: {e}")

    except Exception as e:
        logger.error(e)
    finally:
        if con:
            con.close()

    # Create a DataFrame from the summary data
    summary_df = pd.DataFrame(summary_data)

    # Set the table name as the index
    if not summary_df.empty:
        summary_df = summary_df.set_index("Table")
    return summary_df


def save_summary_to_db(db_path, summary_df):
    """
    Saves the summary DataFrame to the strategy_decisions.db as a table named 'summary'.

    Args:
        db_path (str): The path to the strategy_decisions.db file.
        summary_df (pd.DataFrame): The summary DataFrame to save.
    """
    logger = setup_logging("logs", "summary.log", level=logging.INFO)
    try:
        con = sqlite3.connect(db_path)
        if not summary_df.empty:
            summary_df.to_sql("summary", con, if_exists="replace", index=True)
            logger.info("Summary DataFrame saved to 'summary' table in the database.")
        else:
            logger.warning("Summary DataFrame is empty. Nothing to save.")
    except Exception as e:
        logger.error(f"Error saving summary to database: {e}")
    finally:
        if con:
            con.close()


def summarize_trades_list(db_path: str) -> pd.DataFrame:
    """
    Summarize the data in the trades_list.db SQLite database.

    Parameters:
    db_path (str): Path to the trades_list.db SQLite database.

    Returns:
    pd.DataFrame: A summary DataFrame containing metrics for each strategy.
    """
    try:
        # Connect to the database
        con = sqlite3.connect(db_path)

        # Get the list of tables (strategies) in the database
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql(query, con)["name"].tolist()
        # Remove 'summary' from the list if it exists
        if "summary" in tables:
            tables.remove("summary")

        # Initialize a list to store summary data for each strategy
        summary_data = []

        for table in tables:
            # Load the data for the current strategy
            df = pd.read_sql(f"SELECT * FROM {table}", con)

            # Calculate summary metrics
            total_trades = len(df)
            successful_trades = df[df["ratio"] > 1].shape[0]
            failed_trades = df[df["ratio"] <= 1].shape[0]
            pct_successful = (successful_trades / total_trades) * 100
            ratio_avg = df["ratio"].mean()
            buy_price_sum = df["buy_price"].sum()
            current_price_sum = df["current_price"].sum()
            profit = current_price_sum - buy_price_sum

            # portfolio_value = df["portfolio_value"].iloc[-1] if not df.empty else 0

            # Append the summary for the current strategy
            summary_data.append(
                {
                    "strategy": table,
                    "total_trades": total_trades,
                    "successful_trades": successful_trades,
                    "failed_trades": failed_trades,
                    "pct_successful": round(pct_successful, 1),
                    "ratio_avg": round(ratio_avg, 2),
                    "profit": round(profit, 2),
                    # "portfolio_value": portfolio_value,
                }
            )

        # Convert the summary data to a DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(by="pct_successful", ascending=False)
        summary_df = summary_df.reset_index(drop=True)

        return summary_df

    except Exception as e:
        print(f"Error summarizing trades list: {e}")
        return pd.DataFrame()

    finally:
        con.close()


if __name__ == "__main__":

    strategy_decisions_db_path = os.path.join(
        "PriceData", "strategy_decisions_final.db"
    )
    summary_df = summarize_strategy_decisions(strategy_decisions_db_path)
    save_summary_to_db(strategy_decisions_db_path, summary_df)

    # trades_list_db_name = os.path.join("PriceData", "trades_list.db")
    # summary_df = summarize_trades_list(trades_list_db_name)
    # save_summary_to_db(trades_list_db_name, summary_df)
    print(summary_df)
