from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
from financial_models import train_and_extract_params
from database_retrieval import (
    fetch_price_history,
    save_model_results,
    get_latest_metrics,
)

app = FastAPI(title="Pull-Model Training Service")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchTrainingRequest(BaseModel):
    symbols: List[str]


def process_batch_logic(symbols: List[str]):
    """
    The core loop: Fetch -> Train -> Store -> Repeat.
    This runs in the background to avoid blocking the HTTP response.
    """
    results_summary = []

    for symbol in symbols:
        prices = fetch_price_history(symbol)

        if not prices:
            logger.warning(f"Skipping {symbol}: No price data found.")
            continue

        # Ensure we have enough data points to train (returns need at least a few observations)
        if len(prices) < 10:
            logger.warning(
                f"Skipping {symbol}: Not enough price points ({len(prices)}) to train models."
            )
            continue

        model_result = train_and_extract_params(prices)

        if model_result.get("success"):
            save_model_results(symbol, model_result)
            results_summary.append({symbol: "Success"})
        else:
            logger.error(f"Training failed for {symbol}: {model_result.get('error')}")

    logger.info(f"Batch complete. Processed {len(results_summary)} symbols.")


def process_batch_logic_sync(symbols: List[str]):
    """
    The core loop: Fetch -> Train -> Store -> Repeat.
    Returns results instead of using background tasks or saving to DB.
    """
    all_metrics = []

    for symbol in symbols:
        prices = fetch_price_history(symbol)

        if not prices or len(prices) < 10:
            logger.warning(f"Skipping {symbol}: Insufficient price data.")
            continue

        model_result = train_and_extract_params(prices)

        if model_result.get("success"):
            # Add the symbol to the results for easy client parsing
            model_result["symbol"] = symbol
            all_metrics.append(model_result)
        else:
            logger.error(f"Training failed for {symbol}: {model_result.get('error')}")

    logger.info(f"Batch complete. Processed {len(all_metrics)} symbols.")
    return all_metrics


@app.post("/train_sync")
def trigger_training_sync(payload: BatchTrainingRequest):  # Removed BackgroundTasks
    """
    Receives a list of company names, processes them synchronously, and returns the results.
    """
    if not payload.symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    results_summary = process_batch_logic_sync(payload.symbols)

    return {
        "status": "Processing complete",
        "results": results_summary,
    }


@app.post("/train_batch")
def trigger_training(payload: BatchTrainingRequest, background_tasks: BackgroundTasks):
    """
    Receives a list of company names.
    Immediately returns 'Accepted' while the heavy lifting happens in the background.
    """
    if not payload.symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    background_tasks.add_task(process_batch_logic, payload.symbols)

    return {
        "status": "Batch processing started",
        "message": f"Queued {len(payload.symbols)} symbols for training.",
        "mode": "Pull Architecture",
    }


@app.get("/results/{symbol}")
def get_model_results(symbol: str):
    """
    Retrieves the latest trained model parameters for a given symbol.
    """
    data = get_latest_metrics(symbol)

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No model data found for {symbol}.",
        )

    return {"status": "success", "data": data}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
