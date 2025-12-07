import requests
import json
import csv
import logging
from typing import List, Dict, Any

# --- Configuration ---
SYMBOLS_CSV_PATH = "symbols.csv"
OUTPUT_JSON_PATH = "model_parameters.json"
API_URL = "http://localhost:8001/train_sync"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_symbols_from_csv(file_path: str) -> List[str]:
    """Reads symbols from a CSV file (assuming 'symbol' column)."""
    symbols = []
    try:
        with open(file_path, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            # Assuming the header is 'symbol'
            symbols = [row["symbol"].strip() for row in reader if row.get("symbol")]
        logger.info(f"Read {len(symbols)} symbols from {file_path}")
        return symbols
    except FileNotFoundError:
        logger.error(f"Symbol file not found at {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading symbols: {e}")
        return []


def fetch_and_save_parameters():
    """Reads symbols, calls the API, and overwrites the JSON file."""

    symbols_list = read_symbols_from_csv(SYMBOLS_CSV_PATH)
    if not symbols_list:
        logger.error("No symbols to process. Exiting.")
        return

    try:
        payload = {"symbols": symbols_list}
        logger.info(f"Sending request to {API_URL} for {len(symbols_list)} symbols...")

        # Make a blocking POST request
        response = requests.post(API_URL, json=payload, timeout=3600)
        response.raise_for_status()

        data = response.json()

        if data.get("status") == "Processing complete" and "results" in data:
            all_parameters = data["results"]

            # for easier lookup (e.g., {'AAPL': {...}, 'MSFT': {...}})
            parameters_dict = {item.pop("symbol"): item for item in all_parameters}

            with open(OUTPUT_JSON_PATH, "w") as f:
                json.dump(parameters_dict, f, indent=4)

            logger.info(
                f"Successfully saved {len(all_parameters)} sets of parameters to {OUTPUT_JSON_PATH}"
            )
        else:
            logger.error(f"API call failed with unexpected status: {data}")

    except requests.exceptions.ConnectionError:
        logger.error(f"Connection failed. Is the server running at {API_URL}?")
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP error occurred: {err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    fetch_and_save_parameters()
