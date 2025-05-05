"""
ONPWebScout.py

Our ONPWebScout program for automatically searching online retailers for emerging ONP brands. 

To use, you must replace the GOOGLE_CLOUD_API_KEY, GOOGLE_SEARCH_ENGINE_ID, and LLM_API_KEY with your own keys/ids. 
"""
#%%
import pandas as pd
from stages.web_search import search
from stages.llm_eval import llm_eval
from stages.consolidation import consolidation

#%% 
# 1. Web Search
# We use Google to search for nicotine pouch stores

## Constants
# Replace with your Google API Key, https://console.cloud.google.com/apis/credentials
GOOGLE_CLOUD_API_KEY = 'XXX' 
# Replace with your search engine ID, https://programmablesearchengine.google.com/controlpanel/all
GOOGLE_SEARCH_ENGINE_ID = 'XXX' 
QUERY = 'nicotine pouch online store'

search_results = search(GOOGLE_CLOUD_API_KEY, GOOGLE_SEARCH_ENGINE_ID, QUERY)

#%%
# 2. LLM Evaluation
# We use ChatGPT to evaluate the search results and determine which are stores and their brands

## Constants ##
# Replace with your LLM API Key. For ChatGPT, https://platform.openai.com/settings/organization/api-keys
LLM_API_KEY = 'XXX'
QUERY = "Answer True/False whether this website is a store that sells nicotine pouches. If True, give a Python list of the brands of nicotine pouches sold. Otherwise, brands are 'NA'. Flavors such as wintergreen, mint, or cool breeze are not brands. If a product is listed as 'VELO Nicotine Pouches', the brand is 'VELO' only. Exclude snus products, we only care about nicotine pouches. Only respond with a Python dict with columns 'is_store' and 'brands', where brands is a Python list. Two examples follow: {'is_store': True, 'brands':['ZYN', 'VELO']}, {'is_store': False, 'brands':'NA'}. "

llm_results = llm_eval(LLM_API_KEY, search_results, QUERY)

#%%
# 3. Consolidation
# We consolidate the brand list and merge with previous list.

## Constants ##
# Set to None or remove this parameter if you have no old brand file
OLD_BRANDS_FILE = pd.read_csv('old_brands.csv')

consolidated_results = consolidation(llm_results, OLD_BRANDS_FILE)

#%%
# Saving the results

## Constants ##
OUTPUT_PATH = 'brands_list_new.csv'

consolidated_results.to_csv(OUTPUT_PATH, index=False)
