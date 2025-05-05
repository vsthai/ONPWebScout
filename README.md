# ONPWebScout

Automatic tool for scraping oral nicotine pouch (ONP) brand data. Our pipeline first performs a Google Search to find retailer websites, then uses ChatGPT to determine which results are stores and extract the brands they carry. Finally, results are consolidated and duplicates removed.

## Features

- Monitors ONP brands across the web
- Automatically detects brands using LLMs
- Consolidates and cleans scraped data

## Folder Structure

Our pipeline consists of 3 stages: web search, LLM evaluation, and result consolidation. These are located in the `stages/` directory.

We also use a modified version of two class from [ScrapeGraphAI](https://github.com/ScrapeGraphAI/Scrapegraph-ai), licensed under the MIT license. The adapted code and the license text are in the `scrapegraphAI_extension/` directory.

## Requirements

- Python 3.12
- List of Python dependencies in `requirements.txt`
- **ScrapeGraphAI** requires Playwright. After installing dependencies, run the following command: `playwright install`

## Usage
Simply run the `ONPWebScout.py` script. 

## Credits

- [ScrapeGraphAI](https://github.com/ScrapeGraphAI/Scrapegraph-ai) – for  web scraping using LLMs and graph logic.
- Developed by Veronica Thai.

## Funding

This project was supported by grant number
U54CA287392 from the National Cancer
Institute and FDA Center for Tobacco Products
(CTP). The content is solely the responsibility of
the authors and does not necessarily represent
the official views of the NIH or the Food and
Drug Administration.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.