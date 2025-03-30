create a function called "create_position_column_vectorized". create df['position']. if col "BBANDS_indicator" changes from Hold or Sell to Buy then col position should be 1. It should remain 1 until "BBANDS_indicator" changes from Buy to Sell, or from hold to Sell then col position should revert to its default value of 0. Should only use pandas vectorized methods. use np.select


create a function called "create_cash_column". the function accepts and returns a dataframe. it creates a col called "cash". the starting value is 10000. when col "position" changes from 0 to 1, subtract col "Close" from cash. when col "position" changes from 1 to 0, add col "Close" to cash. should use pandas vectorized methods.

create a function called "create_cash_holdings_column". should use pandas vectorized methods. the function accepts and returns a dataframe. it creates a col called "cash". the starting value is 10000. when col "position" changes from 0 to 1, subtract col "Close" from cash. when col "position" changes from 1 to 0, add col "Close" to cash. 
The function also creates a col called "holdings". the starting value is 0. when col "position" changes from 0 to 1, add col "Close" to holdings and adjust it each day based on Close. when col "position" changes from 1 to 0, holdings col reverts to 0.

create a function called "create_buy_and_sell_date_column". should use pandas vectorized methods. the function accepts and returns a dataframe. it creates a col called "buy_date". the starting value is "". When col "position" changes from 0 to 1, copy the index date into the "buy_date" col. repeat the buy_date until col "position" changes from 1 to 0.
when col "position" changes from 1 to 0, set the "buy_date" col to "".

also create a col called "sell_date". when position changes from 1 to 0, copy the index date into the "sell_date" col.


create a function called "merge_price_and_regime_data". make the function efficient, should use pandas vectorized methods. There is list of trades stored in sqlite db called 'trades_list_vectorized.db' cols are 'Ticker', 'buy_date' and 'sell_date'.
the function should lookup the prices for the buy and sell dates from another sqlite db called 'price_data.db'. the price data is stored as 1 table for each ticker.

2 lookups are required:
1. lookup the 'Close' price for "buy_date" from price_data.db and add it to a col "buy_price" in the trades_list df
2. lookup the 'Close' price for "sell_date" from price_data.db and add it to a col "sell_price" in the trades_list df

save the result as a df.


create a function called 'prepare_sp500_one_day_return'. make the function efficient, should use pandas vectorized methods. the function will have 1 arguement, which is a sqlite connection to the 'price_data.db'. The function should get the sp500 price data from the table named '^GSPC' as a df. Then it will calculate a new col called '1_day_pct_return' using pandas. Finally save the new df back to the 'price_data.db' replacing the '^GSPC' table.


create a function called "lookup_regime_data". make the function efficient, should use pandas vectorized methods. There is list of trades stored in sqlite db called 'trades_list_vectorized.db' cols are 'Ticker', 'buy_date' and 'sell_date'.
the function should lookup the "^VIX" Close price and "^GSPC" "1_day_spy_return" for the buy dates from another sqlite db called 'price_data.db'. the price data is stored as 1 table for each ticker.

2 lookups are required:
1. lookup the "^VIX" 'Close' price for "buy_date" from price_data.db and add it to a col "^VIX" in the trades_list df
2. lookup the "^GSPC" "1_day_spy_return" value for "buy_date" from price_data.db and add it to a col "1_day_spy_return" in the trades_list df

save the result as a df.