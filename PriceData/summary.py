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


if __name__ == "__main__":
    # Database connection
    db_path = os.path.join("PriceData", "strategy_decisions.db")
    # db_path = "strategy_decisions.db"
    summary_df = summarize_strategy_decisions(db_path)
    print(summary_df)
    save_summary_to_db(db_path, summary_df)
