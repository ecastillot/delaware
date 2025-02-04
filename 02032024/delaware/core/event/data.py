# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2025-01-23 10:54:27
#  * @modify date 2025-01-23 10:54:27
#  * @desc [description]
#  */
import pandas as pd
import copy


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
    # if data.empty:
    #     raise Exception("The data object is empty.")

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


class QuakeDataFrame:
    """
    A subclass of pandas DataFrame to handle data with additional functionalities.

    Attributes:
        data (pd.DataFrame): The processed DataFrame containing data.
        required_columns (list): List of mandatory columns in the DataFrame.
        date_columns (list, optional): List of columns to parse as datetime.
    """

    def __init__(self, data, required_columns, date_columns=None, author=None):
        """
        Initialize the QuakeDataFrame instance.

        Parameters:
            data (pd.DataFrame): Input DataFrame containing earthquake data.
            required_columns (list): List of mandatory columns in the DataFrame.
            date_columns (list, optional): List of columns to parse as datetime. Defaults to None.
            author (str, optional): The author or source of the picks data.
                
        """
        self.data = proc_data(
            data=data, 
            required_columns=required_columns,
            date_columns=date_columns, 
        )

        # Store custom attributes
        self.author = author
        self.required_columns = required_columns
        self.date_columns = date_columns

    @property
    def empty(self):
        """Check if the DataFrame is empty."""
        return self.data.empty

    def __len__(self):
        """Return the number of rows in the DataFrame."""
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value

    def __str__(self, extended=False):
        """
        Return a string representation of the QuakeDataFrame instance.

        Parameters:
            extended (bool): If True, return the full DataFrame as a string. Defaults to False.

        Returns:
            str: String representation of the QuakeDataFrame.
        """
        if extended:
            msg = self.data.__str__()
        else:
            msg = f"QuakeDataFrame ({self.__len__()} rows)"
            # msg += "\n-" * len(msg)
        return msg

    def append(self, data):
        """
        Append new data to the QuakeDataFrame.

        Parameters:
            data (pd.DataFrame): DataFrame to append.

        Returns:
            QuakeDataFrame: Updated QuakeDataFrame instance.
        
        Raises:
            TypeError: If the input data is not a DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            data = proc_data(
                data, 
                required_columns=self.required_columns,
                date_columns=self.date_columns,
            )
            self.data = pd.concat([self.data, data])
        else:
            msg = 'Append only supports a single DataFrame object as an argument.'
            raise TypeError(msg)
        return self

    def remove_data(self, rowval):
        """
        Remove rows from the data based on specified conditions.

        Parameters:
            rowval (dict): Dictionary where keys are column names and values are lists of values to remove.

        Returns:
            QuakeDataFrame: Updated QuakeDataFrame instance.
        
        Raises:
            Exception: If `rowval` is not a dictionary.
        """
        if not isinstance(rowval, dict):
            raise Exception("rowval must be a dictionary")
        
        mask = self.data.isin(rowval)
        mask = mask.any(axis='columns')
        self.data = self.data[~mask]
        self.data.reset_index(drop=True, inplace=True)
        return self
    
    def select_data(self, rowval):
        """
        Select rows in the data based on specified criteria.

        Parameters:
        -----------
        rowval : dict
            A dictionary specifying the columns and the values to select.
            Keys represent column names, and values are lists of values to filter by.

        Returns:
        --------
        self : QuakeDataFrame
            The updated QuakeDataFrame with only the selected rows.
        """
        if not isinstance(rowval, dict):
            raise Exception("rowval must be a dictionary")

        if self.empty:
            return self

        # Create a mask based on the specified selection criteria
        mask = self.data.isin(rowval).any(axis="columns")
        self.data = self.data[mask]
        self.data.reset_index(drop=True, inplace=True)
        return self

    def copy(self):
        """
        Create a deep copy of the QuakeDataFrame instance.

        Returns:
        --------
        QuakeDataFrame
            A deep copy of the current instance.
        """
        return copy.deepcopy(self)
    
    def sort_values(self, **args):
        """
        Sort the DataFrame by the specified columns.

        Parameters:
        -----------
        args : dict
            Arguments passed to `pd.DataFrame.sort_values`.

        Returns:
        --------
        self : QuakeDataFrame
            The updated QuakeDataFrame instance with sorted data.
        """
        self.data = self.data.sort_values(**args)
        self.data.reset_index(drop=True, inplace=True)
        return self

    def filter(self, key, start=None, end=None):
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
        self : QuakeDataFrame
            The updated QuakeDataFrame instance with filtered rows.
        """
        if (start is not None) and (len(self) != 0):
            self.data = self.data[self.data[key] >= start]
        
        if (end is not None) and (len(self) != 0):
            self.data = self.data[self.data[key] <= end]
        
        self.data.reset_index(drop=True, inplace=True)
        return self
    
class MulQuakeDataFrame:
    """
    A class to manage and process multiple QuakeDataFrame objects as a collection.
    
    Attributes:
    -----------
    datahelpers : list
        A list of QuakeDataFrame objects.
    """

    def __init__(self, datahelpers=[]):
        """
        Initialize the MulQuakeDataFrame class.

        Parameters:
        -----------
        datahelpers : list, optional
            A list of QuakeDataFrame objects to initialize the collection (default is an empty list).
        """
        self.datahelpers = datahelpers
        self.name = "MulQuakeDataFrame"

    @property
    def single_name(self):
        return self.name.replace("Mul", "")
    
    @property
    def required_columns(self):
        return self[0].required_columns

    def __iter__(self):
        """
        Enable iteration over the datahelpers list.

        Returns:
        --------
        iterator
            An iterator over the datahelpers list.
        """
        return iter(self.datahelpers)

    def __nonzero__(self):
        """
        Check if the datahelpers list is not empty.

        Returns:
        --------
        bool
            True if the list has elements, False otherwise.
        """
        return bool(len(self.datahelpers))

    def __len__(self):
        """
        Get the number of QuakeDataFrame objects in the collection.

        Returns:
        --------
        int
            The number of elements in the datahelpers list.
        """
        return len(self.datahelpers)

    def _get_str(self, extended=False, object_name=None) -> str:
        """
        Get string representation of the mulobject class.

        Parameters:
        -----------
        extended : bool, optional
            Whether to include extended details for each object in the mulobject (default is False).
        object_name: str, optional
            Name of the mulobject (default is None)
        
        Returns:
        --------
        str
            A formatted string summarizing the collection and optionally detailed descriptions.
        """
        if object_name is None:
            object_name = self.name
        
        
        msg = f"{object_name} ({len(self)} {self.single_name} objects)\n"
        msg += "-" * len(msg)

        submsgs = []
        for i, dfh in enumerate(self.__iter__(), start=1):
            submsg = f"{i}). {dfh.__str__()}"
            if extended:
                submsg += "\n" + dfh.__str__(extended=extended) + "\n"
            submsgs.append(submsg)

        if len(self.datahelpers) <= 20 or extended is True:
            submsgs = "\n".join(submsgs)
        else:
            first_three = submsgs[:3]
            last_two = submsgs[-2:]
            remaining = len(self.datahelpers) - len(first_three) - len(last_two)
            submsgs = "\n".join(first_three + [f"...{remaining} other {self.single_name} objects..."] + last_two)

        return msg + "\n" + submsgs

    def __str__(self, extended=False) -> str:
        """
        String representation of the MulQuakeDataFrame class.

        Parameters:
        -----------
        extended : bool, optional
            Whether to include extended details for each QuakeDataFrame (default is False).
        Returns:
        --------
        str
            A formatted string summarizing the collection and optionally detailed descriptions.
        """

        return self._get_str(extended=extended)

    def __setitem__(self, index, trace):
        """
        Set an item in the datahelpers list at the specified index.

        Parameters:
        -----------
        index : int
            The index where the item should be set.
        trace : QuakeDataFrame
            The new QuakeDataFrame to replace the existing one at the index.
        """
        self.datahelpers[index] = trace

    def __getitem__(self, index):
        """
        Retrieve an item or slice from the `datahelpers` list.

        Parameters:
        -----------
        index : int, slice, or str
            - If an int, retrieves a single QuakeDataFrame at the specified index.
            - If a slice, returns a new MulQuakeDataFrame with the sliced data.
            - If a str, filters and returns all QuakeDataFrames with the specified author.

        Returns:
        --------
        QuakeDataFrame or MulQuakeDataFrame or list[QuakeDataFrame]
            - A single QuakeDataFrame if `index` is an int.
            - A new MulQuakeDataFrame containing the slice if `index` is a slice.
            - A list of QuakeDataFrames matching the specified author if `index` is a str.
        """
        if isinstance(index, slice):
            return self.__class__(datahelpers=self.datahelpers[index])
        elif isinstance(index, str):
            return [dfh for dfh in self.datahelpers if dfh.author == index]
        else:
            return self.datahelpers[index]

    def __delitem__(self, index):
        """
        Delete an item from the datahelpers list at the specified index.

        Parameters:
        -----------
        index : int
            The index of the item to delete.
        """
        del self.datahelpers[index]

    def __getslice__(self, i, j, k=1):
        """
        Get a slice of the datahelpers list.

        Parameters:
        -----------
        i : int
            Start index.
        j : int
            End index.
        k : int, optional
            Step size (default is 1).

        Returns:
        --------
        MulQuakeDataFrame
            A new MulQuakeDataFrame containing the slice.
        """
        return self.__class__(datahelpers=self.datahelpers[max(0, i):max(0, j):k])

    def remove_data(self, rowval):
        """
        Remove rows from each QuakeDataFrame based on specified conditions.

        Parameters:
        -----------
        rowval : dict
            A dictionary specifying the column names (keys) and values to remove (values).

        Returns:
        --------
        MulQuakeDataFrame
            The updated instance with the rows removed.
        """
        self.datahelpers = [dfh.remove_data(rowval) for dfh in self.datahelpers]
        return self

    def select_data(self, rowval):
        """
        Select rows from each QuakeDataFrame based on specified conditions.

        Parameters:
        -----------
        rowval : dict
            A dictionary specifying the column names (keys) and values to select (values).

        Returns:
        --------
        MulQuakeDataFrame
            The updated instance with the selected rows.
        """
        self.datahelpers = [dfh.select_data(rowval) for dfh in self.datahelpers]
        return self

    def copy(self):
        """
        Create a deep copy of the MulQuakeDataFrame instance.

        Returns:
        --------
        MulQuakeDataFrame
            A new instance with the same data.
        """
        return copy.deepcopy(self)

    def sort_values(self, **args):
        """
        Sort values within each QuakeDataFrame.

        Parameters:
        -----------
        **args
            Parameters to pass to the pd.DataFrame.sort_values method.

        Returns:
        --------
        MulQuakeDataFrame
            The updated instance with sorted data.
        """
        self.datahelpers = [dfh.sort_values(**args) for dfh in self.datahelpers]
        return self

    def filter(self, key, start=None, end=None):
        """
        Filter data within each QuakeDataFrame based on a column and range.

        Parameters:
        -----------
        key : str
            The name of the column to filter.
        start : int, float, or datetime, optional
            The start value for filtering (default is None).
        end : int, float, or datetime, optional
            The end value for filtering (default is None).

        Returns:
        --------
        MulQuakeDataFrame
            The updated instance with filtered data.
        """
        self.datahelpers = [dfh.filter(key, start, end) for dfh in self.datahelpers]
        return self