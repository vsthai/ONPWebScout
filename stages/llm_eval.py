"""
llm_eval.py

Contains code for querying ChatGPT to process search results and for cleaning up the returned results.

Functions:
    query_llm(api_key, search_results, query): Queries ChatGPT for each item in search_results using the given query.
    parse_results(chatgpt_results): Parses the results from query_llm, separating the content dict string into separate columns.
    llm_eval(api_key, search_results, query): Queries ChatGPT for each item in search_results using the given query and parses the results.
"""
import pandas as pd
from datetime import datetime
import ast
from scrapegraphai_extension.ScrapeGraphAI_no_parsing import SmartScraperGraphNoParse
import ast
import re

def query_llm(api_key: str,
              search_results: pd.DataFrame,
              query: str) -> pd.DataFrame:
    """
    Queries ChatGPT to process all entries in search_results using ScrapeGraphAI. 

    Args:
        api_key (str): ChatGPT API key to use.
        search_results (str): DataFrame of Google Search results, with a link column consisting of result links.
        query (str): The query to use with ChatGPT. 
    
    Returns:
        DataFrame of ChatGPT results, with columns Link, Date, and Results.
    """
    links = []
    ymds = []
    results = []
    df = pd.DataFrame(columns=['Link', 'Date', 'Results'])
    for _, row in search_results.iterrows():
        url = row['link']
        links.append(url)

        # Format the date and time as "mm-dd-yyhhmmss"
        ymd = datetime.now().strftime('%Y-%m-%d')
        ymds.append(ymd)

        # Define the configuration for the scraping pipeline
        graph_config = {
            "llm": {
                "api_key": api_key, 
                "model": "openai/gpt-4o",
            },
            "verbose": True,
            "headless": False,
        }

        # Create the SmartScraperGraph instance
        smart_scraper_graph = SmartScraperGraphNoParse(
            prompt=query,
            source=url,
            config=graph_config
        )

        # Run the pipeline
        result = smart_scraper_graph.run()

        results.append(result)

    df['Link'] = links
    df['Date'] = ymds
    df['Results'] = results
    return df

def parse_results(chatgpt_results: pd.DataFrame) -> pd.DataFrame:
    """
    Parses and outputs the results from ChatGPT. 

    Args:
        chatgpt_results (str): DataFrame of ChatGPT results, with a Results column for the results.
    
    Returns:
        DataFrame of parsing results, with column Result parsed to separate columns
    """
    # Manual parsing
    results =  chatgpt_results.copy()
    results_content = results['Results'].apply(lambda x: x.content)
    results_content_clean = results_content.apply(lambda x: x.replace('\n', '')) # Remove newlines
    results_content_clean = results_content_clean.apply(lambda x: re.sub(r'\btrue\b', 'True', x, flags=re.IGNORECASE)) # Standardize Trues
    results_content_clean = results_content_clean.apply(lambda x: re.sub(r'\bfalse\b', 'False', x, flags=re.IGNORECASE)) # Standardize Falses
    
    parsed = results_content_clean.apply(lambda x: ast.literal_eval(x))
    results_df = pd.DataFrame(parsed.to_list())

    results = pd.concat([results, results_df], axis=1)

    results = results.drop(columns=['Results'])
    return results

def llm_eval(api_key: str,
             search_results: pd.DataFrame,
             query: str) -> pd.DataFrame:
    """
    Queries ChatGPT and parses the results. 

    Args:
        api_key (str): ChatGPT API key to use.
        search_results (str): DataFrame of Google Search results, with a link column consisting of result links.
        query (str): The query to use with ChatGPT. 

    Returns:
        DataFrame of parsed ChatGPT results.
    """
    llm_output = query_llm(api_key, search_results, query)
    parsing_output = parse_results(llm_output)

    return parsing_output
