import sqlite3
import pandas as pd
import datetime as dt

def save_to_sqlite(df, db_path, table_name):
    """
    Save a DataFrame to an SQLite database, appending data if the table exists.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        db_path (str): The path to the SQLite database file.
        table_name (str): The name of the table in the SQLite database.

    Notes:
        - If the table already exists in the database, new data will be appended.
        - The DataFrame's index will not be saved to the database.
    """
    with sqlite3.connect(db_path) as conn:
        # Save the DataFrame to the SQLite database
        # If the table exists, new rows are appended; otherwise, a new table is created
        df.to_sql(table_name, conn, if_exists='append', index=False)

def load_from_sqlite(
    db_path, tables=None, custom_params=None,
    parse_dates=None, drop_duplicates=True, sortby=None,
    debug=True
):
    """
    Load a DataFrame from an SQLite database based on optional query parameters.

    Args:
        db_path (str): Path to the SQLite database file.
        tables (list of str, optional): List of table names to load data from. 
            If None, loads data from all tables. Defaults to None.
        custom_params (dict, optional): Custom filtering parameters for the query. 
            Format: {column_name: {'value': value, 'condition': condition}}.
            Example for filtering stations starting with 'OKAS' and distance < 0.5:
                custom_params = {
                    "distance": {"condition": "<", "value": 0.5},
                    "station": {"condition": "LIKE", "value": "OKAS%"}
                }
            Defaults to None.
        parse_dates (list of str, optional): List of columns to parse as datetime. Defaults to None.
        drop_duplicates (bool, optional): Whether to drop duplicate rows. Defaults to True.
        sortby (str, optional): Column name to sort the resulting DataFrame by. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing data from the specified table(s) filtered
            based on the provided parameters.

    Raises:
        Exception: If `custom_params` is not structured correctly.

    Notes:
        - If no tables are specified, data from all database tables is loaded.
        - DataFrame is sorted by `sortby` column if provided.
        - Duplicates are removed if `drop_duplicates` is True.
    """
    with sqlite3.connect(db_path) as conn:
        # Get the list of tables in the database
        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
        
        all_tables = pd.read_sql_query(tables_query, conn)['name'].tolist()
        
        if tables is None:
            tables = all_tables # Only use the specified table
        else:
            complement = list(set(tables).difference(all_tables))
            tables = list(set(tables).intersection(all_tables))
            if debug:
                print(f"Number of tables  found: {len(tables)} ")
                print(f"Number of tables not found: {len(complement)} ")

        # Load data from the specified tables
        return load_table(
            conn, table_names=tables, custom_params=custom_params,
            parse_dates=parse_dates, drop_duplicates=drop_duplicates, sortby=sortby
        )


def load_chunks_from_sqlite(
    db_path, chunksize=None, custom_params=None,
    parse_dates=None, drop_duplicates=True, sortby=None
):
    """
    Load a DataFrame from an SQLite database in chunks.

    Args:
        db_path (str): Path to the SQLite database file.
        chunksize (int, optional): Number of rows to load per chunk. Defaults to None.
        custom_params (dict, optional): Custom filtering parameters for the query. Same structure as above.
        parse_dates (list of str, optional): List of columns to parse as datetime. Defaults to None.
        drop_duplicates (bool, optional): Whether to drop duplicate rows. Defaults to True.
        sortby (str, optional): Column name to sort the resulting DataFrame by. Defaults to None.

    Yields:
        pd.DataFrame: DataFrame chunks filtered and sorted as specified.

    Notes:
        - Use this function for large datasets to avoid memory issues.
    """
    with sqlite3.connect(db_path) as conn:
        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
        chunk_iterator = pd.read_sql_query(tables_query, conn, chunksize=chunksize)

        # Iterate over table chunks
        for chunk in chunk_iterator:
            table_names = chunk['name'].tolist()
            yield load_table(
                conn, table_names, custom_params=custom_params,
                parse_dates=parse_dates, drop_duplicates=drop_duplicates, sortby=sortby
            )


def load_table(
    conn, table_names, custom_params=None, parse_dates=None,
    drop_duplicates=True, sortby=None
):
    """
    Load data from specified tables in the SQLite database.

    Args:
        conn (sqlite3.Connection): Connection to the SQLite database.
        table_names (list of str): List of table names to query.
        custom_params (dict, optional): Filtering parameters. See examples above.
        parse_dates (list of str, optional): Columns to parse as datetime. Defaults to None.
        drop_duplicates (bool, optional): Whether to drop duplicate rows. Defaults to True.
        sortby (str, optional): Column name to sort the DataFrame by. Defaults to None.

    Returns:
        pd.DataFrame: A combined DataFrame containing data from the specified tables.

    Notes:
        - Ensures filtering conditions are only applied to valid columns in the table.
        - Validates the structure of `custom_params`.
    """
    
    all_dataframes = []

    for table_name in table_names:
        try:
            # Check column information for the table
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
        except sqlite3.OperationalError:
            print(f"Table '{table_name}' not found in the database.")
            continue

        # Extract column names from the table
        columns = [col[1] for col in cursor.fetchall()]

        # Build query for the table
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        sql_params = {}
        req_keys = ["value", "condition"]

        # Add custom filtering parameters to the query
        if custom_params:
            for key, info in custom_params.items():
                # Validate the structure of custom_params
                for req_key in req_keys:
                    if req_key not in info:
                        raise Exception(
                            "custom_params must follow this structure: "
                            "{x: {'value': y, 'condition': y}}"
                        )

                # Add conditions for columns that exist in the table
                if key in columns:
                    
                    query += f" AND {key} {info['condition']} :{key}"
                    
                    value = info["value"]
                    if isinstance(value, dt.datetime):
                        value = value.strftime("%Y-%m-%d %H:%M:%S")
                        

                
                    sql_params[key] = value
                    
        # Execute the query and load the data into a DataFrame
        df = pd.read_sql_query(query, conn, params=sql_params, parse_dates=parse_dates)

        if df.empty:
            continue

        # Remove duplicates if required
        if drop_duplicates:
            drop_subset = list(custom_params.keys()) if custom_params else None
            df = df.drop_duplicates(subset=drop_subset, ignore_index=True)

        if "arrival_time" in list(df.columns.to_list()):
            df = df.rename(columns={"arrival_time":"time"})  
            
        # print(df.info())
        # Sort the DataFrame if a sort column is specified
        if sortby:
            df = df.sort_values(by=sortby, ignore_index=True)

        # Add the DataFrame to the list
        all_dataframes.append(df)


    # Combine all DataFrames into one
    if all_dataframes:
        return pd.concat(all_dataframes, ignore_index=True)
    else:
        return pd.DataFrame()
            

if __name__ == "__main__":
    path = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_database/TX.PB5.00.CH_ENZ.db"
    # path = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_database/4O.WB10.00.HH_ENZ.db"
    df = load_dataframe_from_sqlite(path, "availability", 
                                    starttime="2024-01-01 00:00:00", 
                                    endtime="2024-08-01 00:00:00")
    print(df)
    
    import sqlite3

    # def list_tables(db_path):
    #     """List all tables in the SQLite database."""
    #     with sqlite3.connect(db_path) as conn:
    #         cursor = conn.cursor()
    #         cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #         tables = cursor.fetchall()
    #         print(tables)
    #         for table in tables:
    #             print(table[0])

    # # Example usage
    # list_tables(path)