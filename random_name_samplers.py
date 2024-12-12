import numpy as np
import typing
import re
import pandas as pd
import random


#random name generator class
class RandomNameGenerator:
    def __init__(self, first_names_df: pd.DataFrame, last_names_df: pd.DataFrame, random_state  : int = 42):
        """

        Args:
          first_names_df: Dataframe with columns [First_name, count] (case-insensitive)
          last_names_df: DataFrame with columns [Last_name,count] (case-insensitive)
          random_state: int
        """

        #set random seed
        random.seed(random_state)

        # Standardize column names to lowercase for case-insensitive access
        first_names_df.columns = [col.lower() for col in first_names_df.columns]
        last_names_df.columns = [col.lower() for col in last_names_df.columns]

        # Check if required columns are present
        if not {'first_name', 'count'}.issubset(first_names_df.columns):
            raise ValueError("first_names_df must contain 'first_name' and 'count' columns")
        if not {'last_name', 'count'}.issubset(last_names_df.columns):
            raise ValueError("last_names_df must contain 'last_name' and 'count' columns")

        # Ensure frequencies are normalized
        first_names_df['frequency'] = first_names_df['count'] / first_names_df['count'].sum()
        last_names_df['frequency'] = last_names_df['count'] / last_names_df['count'].sum()

        # Create cumulative frequency columns for weighted random selection
        first_names_df['cumulative_frequency'] = first_names_df['frequency'].cumsum()
        last_names_df['cumulative_frequency'] = last_names_df['frequency'].cumsum()

        # Store dataframes for use in name generation
        self.first_names_df = first_names_df
        self.last_names_df = last_names_df

    def _random_first_name(self):
        # Select a random value between 0 and 1 for cumulative frequency
        random_value = random.random()
        # Find the first name whose cumulative frequency is just above the random value
        first_name = self.first_names_df.loc[self.first_names_df['cumulative_frequency'] >= random_value].iloc[0]
        return first_name['first_name']

    def _random_last_name(self):
        # Select a random value between 0 and 1 for cumulative frequency
        random_value = random.random()
        # Find the last name whose cumulative frequency is just above the random value
        last_name = self.last_names_df.loc[self.last_names_df['cumulative_frequency'] >= random_value].iloc[0]
        return last_name['last_name']

    def reset_seed(self, seed :int):
        random.seed(seed)

    def get_first_name_last_name_string(self) -> str:
        # Combine random first and last names
        first_name = self._random_first_name()
        last_name = self._random_last_name()
        return " ".join([first_name,last_name])


class RandomPlaceGenerator:
    def __init__(self, places : pd.DataFrame, random_state  : int = 42):
        """
        Args:
          first_names_df: Dataframe with columns ['name']
          random_state: int
        """

        assert places.columns == ['name'], "places must have a column named 'name'"

        random.seed(random_state)
        self.places = places['name'].tolist()
        assert all([isinstance(place,str) for place in self.places]), "all places must be strings"


    def get_place_string(self) -> str:
        return random.choice(self.places)


class RandomOrganisationGenerator:
    def __init__(self, organisations : pd.DataFrame, random_state  : int = 42):
        """

        Args:
          first_names_df: Dataframe with columns [name]
          random_state: int
        """
        assert organisations.columns == ['name'], "places must have a column named 'name'"

        random.seed(random_state)
        self.organisations = organisations['name'].tolist()
        assert all([isinstance(organisation,str) for organisation in self.organisations]), "all organisations must be strings"


    def get_organisation_string(self) -> str:
        return random.choice(self.organisations)