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

def load_dataframe_from_sqlite(db_name, tables=None, starttime=None, endtime=None,
                               debug=True):
    """
    Load a DataFrame from an SQLite database based on optional query parameters.

    Args:
        db_name (str): The path to the SQLite database file.
        table_name (str, optional): The name of the table to load data from. If None, load data from all tables. Defaults to None.
        starttime (str, optional): The start time for the data query in 'YYYY-MM-DD HH:MM:SS' format. Defaults to None.
        endtime (str, optional): The end time for the data query in 'YYYY-MM-DD HH:MM:SS' format. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing data from the specified table(s) and time range.

    Notes:
        - The `starttime` and `endtime` parameters are optional. The query will only filter by these fields if they exist in the table(s) and are provided.
        - The DataFrame's 'starttime' and 'endtime' columns are converted to datetime objects if they exist.
        - The DataFrame is sorted by 'starttime' after loading if the column exists.
    """
    
    with sqlite3.connect(db_name) as conn:
        # Get the list of tables in the database
        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
        
        all_tables = pd.read_sql_query(tables_query, conn)['name'].tolist()
        
        # print(tables)
        # exit()

        if tables is None:
            tables = all_tables # Only use the specified table
        else:
            tables = list(set(tables).intersection(all_tables))
            complement = list(set(all_tables).difference(tables))
            if debug:
                print(f"{len(complement)} tables not found")
            
        # If no table is specified, load from all tables
        all_dataframes = []

        for table in tables:
            # Get the list of columns in the current table
            try:
                cursor = conn.execute(f"PRAGMA table_info({table})")
            except:
                if debug:
                    print(f"Not found {table}")
                continue
            
            columns = [col[1] for col in cursor.fetchall()]

            # Build query for the current table
            query = f"SELECT * FROM {table} WHERE 1=1"

            # Add conditions for starttime and endtime if the columns exist and parameters are provided
            params = []
            if 'starttime' in columns and starttime:
                query += " AND starttime >= ?"
                params.append(starttime)
            if 'endtime' in columns and endtime:
                query += " AND endtime <= ?"
            
            # Execute the query for the current table
            df = pd.read_sql_query(query, conn, params=params)

            # Convert 'starttime' and 'endtime' columns to datetime if they exist
            if 'starttime' in df.columns:
                df['starttime'] = pd.to_datetime(df['starttime'])
            if 'endtime' in df.columns:
                df['endtime'] = pd.to_datetime(df['endtime'])

            # Drop duplicates based on 'starttime' and 'endtime' if they exist
            drop_subset = [col for col in ['starttime', 'endtime'] if col in df.columns]
            if drop_subset:
                df = df.drop_duplicates(subset=drop_subset, ignore_index=True)

            # Sort DataFrame by 'starttime' if it exists
            if 'starttime' in df.columns:
                df = df.sort_values(by=['starttime'], ignore_index=True)

            all_dataframes.append(df)

        if all_dataframes:
            # Concatenate all DataFrames from all tables
            df = pd.concat(all_dataframes, ignore_index=True)
        else:
            df = pd.DataFrame()
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