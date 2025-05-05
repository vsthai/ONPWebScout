# Brand List Format

We explain the format of the brand list and provide a sample file, [sample.csv](./sample.csv).

## The `brand` column
The `brand` column consists of a single brand name OR a Python list of brands that correspond to the same brand. For example, "FRE" and "FRĒ" refer to the same brand, so the `brand` column entry is: `['FRE',  'FRĒ']`. Brands are treated as case-insensitive.

## The `stores` column
The `stores` column consists of Python lists of links to stores that carry a particular brand. 
