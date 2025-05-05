"""
web_search.py

Contains code for making Google searches using their API. 

Functions:
    search(api_key, search_engine_id, query, num_results): Makes the search using the given query and returns the results.
"""

import requests
import pandas as pd

def search(api_key: str,
           search_engine_id: str,
           query: str,
           num_results: int = 25,
           ) -> pd.DataFrame:
    """
    Uses the Google Search API to find the top results for a given query and output the results to a CSV file. 

    Args:
        api_key (str): Google API key to use.
        search_engine_id (str): Google Search engine ID to use (enables custom search).
        query (str): Search query to use.
        num_results (int, optional): Number of search results to return. Defaults to 25.
    
    Returns:
        DataFrame of search results, with columns title, link, and snippet.
    """

    # Google Custom Search API URL
    url = 'https://www.googleapis.com/customsearch/v1'

    index = 1
    search_results = []

    # Need to do multiple passes, Google limits to 10 results
    while index <= num_results:
        num_this_pass = min(num_results - index + 1, 10)

        # Parameters for the search
        params = {
            'key': api_key,
            'cx': search_engine_id,
            'q': query,
            'num': num_this_pass,
            'start': index
        }

        # Send the request
        response = requests.get(url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            results_this_pass = response.json()
            search_results.extend(results_this_pass['items'])

        else:
            print(f'Error {index} - {index + num_this_pass}:', response.status_code)

        index = index + num_this_pass

    search_results = pd.DataFrame(search_results)[['title', 'link', 'snippet']]
    return search_results
