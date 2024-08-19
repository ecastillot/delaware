import sqlite3
import pandas as pd

def save_dataframe_to_sqlite(df, db_name, table_name):
    """
    Save a DataFrame to an SQLite database, appending data if the table exists.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        db_name (str): The path to the SQLite database file.
        table_name (str): The name of the table in the SQLite database.

    Notes:
        - If the table already exists in the database, new data will be appended.
        - The DataFrame's index will not be saved to the database.
    """
    with sqlite3.connect(db_name) as conn:
        # Save DataFrame to SQLite database, appending if the table exists
        df.to_sql(table_name, conn, if_exists='append', index=False)

def load_dataframe_from_sqlite(db_name, table_name, starttime, endtime):
    """
    Load a DataFrame from an SQLite database based on query parameters.

    Args:
        db_name (str): The path to the SQLite database file.
        table_name (str): The name of the table to load data from.
        starttime (str): The start time for the data query in 'YYYY-MM-DD HH:MM:SS' format.
        endtime (str): The end time for the data query in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
        pd.DataFrame: A DataFrame containing data from the specified table and time range.

    Notes:
        - The `starttime` and `endtime` parameters should be in a format recognized by SQLite.
        - The DataFrame's 'starttime' and 'endtime' columns are converted to datetime objects.
        - The DataFrame is sorted by 'starttime' after loading.
    """
    query = f"""
    SELECT * FROM {table_name}
    WHERE starttime >= ? AND endtime <= ?
    """
    
    with sqlite3.connect(db_name) as conn:
        # Load data from SQLite database based on the query
        df = pd.read_sql_query(query, conn, params=(starttime, endtime))

        # Convert 'starttime' and 'endtime' columns to datetime
        df['starttime'] = pd.to_datetime(df['starttime'])
        df['endtime'] = pd.to_datetime(df['endtime'])

        df = df.drop_duplicates(subset=["starttime","endtime"],ignore_index=True)

        # Sort DataFrame by 'starttime'
        df = df.sort_values(by=["starttime"], ignore_index=True)

    return df

if __name__ == "__main__":
    path = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_database/TX.PB5.00.CH_ENZ.db"
    # path = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_database/4O.WB10.00.HH_ENZ.db"
    df = load_dataframe_from_sqlite(path, "availability", 
                                    starttime="2024-01-01 00:00:00", 
                                    endtime="2024-08-01 00:00:00")
    print(df)
    
    import sqlite3

    # def list_tables(db_name):
    #     """List all tables in the SQLite database."""
    #     with sqlite3.connect(db_name) as conn:
    #         cursor = conn.cursor()
    #         cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #         tables = cursor.fetchall()
    #         print(tables)
    #         for table in tables:
    #             print(table[0])

    # # Example usage
    # list_tables(path)