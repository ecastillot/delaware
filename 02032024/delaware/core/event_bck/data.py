# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2025-01-23 10:54:27
#  * @modify date 2025-01-23 10:54:27
#  * @desc [description]
#  */
import pandas as pd
import copy
import warnings



def proc_data(data, required_columns, date_columns=None):
    """
    Process the input DataFrame by validating columns, removing duplicates, 
    and optionally parsing date information.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing data to process.
        required_columns (list): List of mandatory columns that must be present in the DataFrame.
        date_columns (list, optional): List of columns to be parsed as datetime. Defaults to None.
        

    Returns:
        pd.DataFrame: Processed DataFrame.
    
    Raises:
        Exception: If required columns are missing, if data is empty, 
                   or if invalid parameters are provided.
    """
    # Error message for missing required columns
    msg = {"required_columns": "The mandatory columns are missing in the data object. "
                               + f"Required columns: {required_columns}"}
    # Check if all mandatory columns are present in the DataFrame
    if not all(item in data.columns for item in required_columns):
        raise Exception(msg["required_columns"])

    # Remove duplicate rows based on the required columns
    data.drop_duplicates(subset=required_columns, ignore_index=True, inplace=True)

    # Check if the DataFrame is empty after removing duplicates
    if data.empty:
        raise Exception("The data object is empty.")

    # Parse date columns, if specified
    if date_columns is not None:
        if not isinstance(date_columns, list):
            raise Exception("The 'date_columns' parameter must be a list.")
        for col_date in date_columns:
            if col_date in data.columns:
                data[col_date] = pd.to_datetime(
                    data[col_date], errors="coerce"
                ).dt.tz_localize(None)

    return data


class DataFrameHelper(pd.DataFrame):
    """
    A subclass of pandas DataFrame to handle data with additional functionalities.

    Attributes:
        data (pd.DataFrame): The processed DataFrame containing data.
        required_columns (list): List of mandatory columns in the DataFrame.
        date_columns (list, optional): List of columns to parse as datetime.
    """
    _preprocessed = False  # Class-level flag to track preprocessing
    
    def __init__(self, *args,required_columns=None, date_columns=None, author=None,**kwargs):
        """
        Initialize the DataFrameHelper instance.

        Parameters:
            data (pd.DataFrame): Input DataFrame containing earthquake data.
            required_columns (list): List of mandatory columns in the DataFrame.
            date_columns (list, optional): List of columns to parse as datetime. Defaults to None.
            author (str, optional): The author or source of the picks data.
                
        """
        if not DataFrameHelper._preprocessed:
            if required_columns is not None:
                args = list(args)
                arg_0 = proc_data(args[0],required_columns=required_columns,
                          date_columns=date_columns)
                args = [arg_0]+[x for x in args[1:]]
                args = tuple(args)
                
            DataFrameHelper._preprocessed = True
            
        super().__init__(*args,**kwargs)
        
        warnings.simplefilter("ignore", category=UserWarning)
        self._custom_info = {"required_columns":required_columns,
                             "date_columns": date_columns,
                             "author": author
                             }
        self._instanced = True
            
        
    @property
    def _constructor(self):
        """
        Ensures that operations on this class return instances of DataFrameHelper.
        """
        return DataFrameHelper
    
    def __str__(self, mode="pandas"):
        """
        Return a string representation of the DataFrameHelper instance.

        Parameters:
            extended (bool): If True, return the full DataFrame as a string. Defaults to False.

        Returns:
            str: String representation of the DataFrameHelper.
        """
        if mode == "pandas":
            msg = super().__str__()
        elif mode == "utdquake":
            msg = f"DataFrameHelper ({self.__len__()} rows)" 
        else:
            raise Exception(f"__str__ mode: {mode} is not supported")
            
        return msg
    
    def select_data(self, rowval,inplace=False):
        """
        Select rows in the data based on specified criteria.

        Parameters:
        -----------
        rowval : dict
            A dictionary specifying the columns and the values to select.
            Keys represent column names, and values are lists of values to filter by.

        Returns:
        --------
        self : DataFrameHelper
            The updated DataFrameHelper with only the selected rows.
        """
        
        if not isinstance(rowval, dict):
            raise Exception("rowval must be a dictionary")

        if self.empty:
            return self

        data = self
        # Create a mask based on the specified selection criteria
        mask = data.isin(rowval).any(axis="columns")
        if inplace:
            # Modify the current instance
            self.__init__(data[mask])
            return None
        else:
            # Return a new instance
            return self.__class__(data[mask])
            
    def append(self, data,inplace=False):
        """
        Append new data to the DataFrameHelper.

        Parameters:
            data (pd.DataFrame): DataFrame to append.

        Returns:
            DataFrameHelper: Updated DataFrameHelper instance.
        
        Raises:
            TypeError: If the input data is not a DataFrame.
        """
        data = pd.concat([self, data])
        
        if inplace:
            # Modify the current instance
            self.__init__(data)
            return None
        else:
            # Return a new instance
            return self.__class__(data)
        
    def remove_data(self, rowval,inplace=False):
        """
        Remove rows from the data based on specified conditions.

        Parameters:
            rowval (dict): Dictionary where keys are column names and values are lists of values to remove.

        Returns:
            DataFrameHelper: Updated DataFrameHelper instance.
        
        Raises:
            Exception: If `rowval` is not a dictionary.
        """
        if not isinstance(rowval, dict):
            raise Exception("rowval must be a dictionary")
        
        mask = self.isin(rowval)
        mask = mask.any(axis='columns')
        data = self[~mask]
        
        if inplace:
            # Modify the current instance
            self.__init__(data)
            return None
        else:
            # Return a new instance
            return self.__class__(data)
    
    def filter(self, key, start=None, end=None,inplace=False):
        """
        Filter data in the catalog based on a range of values for a specified column.

        Parameters:
        -----------
        key : str
            Name of the column to filter.
        start : int, float, or datetime.datetime, optional
            The minimum value for the filter range. Must match the data type of `data[key]`.
        end : int, float, or datetime.datetime, optional
            The maximum value for the filter range. Must match the data type of `data[key]`.

        Returns:
        --------
        self : DataFrameHelper
            The updated DataFrameHelper instance with filtered rows.
        """
        data = self
        
        if (start is not None) and (len(data) != 0):
            data = data[data[key] >= start]
        
        if (end is not None) and (len(data) != 0):
            data = data[data[key] <= end]
        
        if inplace:
            # Modify the current instance
            self.__init__(data)
            return None
        else:
            # Return a new instance
            return self.__class__(data)
    