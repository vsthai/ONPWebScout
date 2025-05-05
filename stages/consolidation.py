"""
consolidation.py

Code for consolidating the new brand list with the previous list and removes duplicates.

Functions:
    consolidation(llm_results, old_brands): Combines the llm_results with old_brands (if available) and removes duplicate brands. 
"""
import pandas as pd
import ast
import numpy as np

def consolidation(llm_results: pd.DataFrame, 
                  old_brands: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidates the LLM results removing duplicates, merging with the old brand list, if available, to create the new brand list.

    Args:
        llm_results (str): DataFrame of parsed ChatGPT results, with columns Link, is_store, and brands.
        old_brands (str, optional): DataFrame of old brand list to merge with.

    Returns:
        DataFrame of consolidated results, with columns brand and stores
    """
    # Get the known brands from the old_brands file if available
    known_brands = []
    data = pd.DataFrame(columns=['brand', 'stores'])
    if old_brands is not None:
        known_brands = old_brands['brand'].tolist()
        known_brands = [ast.literal_eval(x) if '[' in x else x for x in known_brands]
        data = old_brands.copy()
        data['stores'] = data['stores'].apply(lambda x: ast.literal_eval(x))

    # Compare each new brand against the old
    for i, row in llm_results.iterrows():
        if not np.isnan(row['is_store']) and row['is_store']:
            brands = row['brands']
            for brand in brands:
                # Checking if this is a known brand.
                # append = -1 for brand not found (new brand), append = -2 for found as string append >= 0 for brand found, and append = index where found
                append = -1
                for i in range(len(known_brands)):
                    known = known_brands[i]

                    if (type(known) == str and known.lower() == brand.lower()) or (type(known) == list and brand.lower() in [x.lower() for x in known]):
                        append = i
                        break

                # If this is a new brand, add it to the known brand list we check against and update the data
                if append == -1: 
                    known_brands.append(brand)
                    data = pd.concat([data, pd.DataFrame([{'brand':brand, 'stores':[row['Link']]}])])
                    data = data.reset_index(drop=True)
                else:
                    # Otherwise, add this store to the stores list
                    data.loc[i, 'stores'].append(row['Link'])

    # Remove duplicate stores
    data['stores'] = data['stores'].apply(lambda x: list(set(x)))

    return data
