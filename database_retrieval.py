import clickhouse_connect
from datetime import datetime
from typing import List, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", 8123))


def get_db_client():
    return clickhouse_connect.get_client(host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT)


def fetch_price_history(symbol: str) -> List[float]:
    """
    Retrieves historical price data for a specific symbol.
    """
    try:
        client = get_db_client()
        query = f"SELECT price FROM stock_prices WHERE symbol = '{symbol}' ORDER BY timestamp ASC"
        result = client.query(query)

        prices = [row[0] for row in result.result_rows]
        return prices

    except Exception as e:
        logger.error("Error fetching data for %s: %s", symbol, e)
        return []


def save_model_results(symbol: str, metrics: Dict[str, Any]):
    """
    Persists the calculated ARMA/GARCH parameters back to ClickHouse.
    """
    try:
        client = get_db_client()

        # Support nested and flattened metric formats like mock DB
        arma = metrics.get("arma", {}) if isinstance(metrics, dict) else {}
        garch = metrics.get("garch", {}) if isinstance(metrics, dict) else {}

        ar_coeff = (
            metrics.get("ar_coeff")
            or arma.get("ar_coef")
            or arma.get("ar_coeff")
            or 0.0
        )
        ma_coeff = (
            metrics.get("ma_coeff")
            or arma.get("ma_coef")
            or arma.get("ma_coeff")
            or 0.0
        )
        garch_volatility = (
            metrics.get("garch_volatility")
            or garch.get("last_volatility")
            or garch.get("garch_volatility")
            or 0.0
        )

        # Structure: [timestamp, symbol, ar_param, ma_param, volatility]
        row = [
            datetime.now(),
            symbol,
            ar_coeff,
            ma_coeff,
            garch_volatility,
        ]

        client.insert(
            table="model_metrics",
            data=[row],
            column_names=["timestamp", "symbol", "ar_param", "ma_param", "volatility"],
        )
        logger.info("Stored metrics for %s in ClickHouse.", symbol)

    except Exception as e:
        logger.error("Error saving model results for %s: %s", symbol, e)


def get_latest_metrics(symbol: str) -> Dict[str, Any]:
    """
    Fetches the most recent ARMA/GARCH parameters for a symbol.
    """
    try:
        client = get_db_client()
        query = f"""
            SELECT ar_param, ma_param, volatility, timestamp 
            FROM model_metrics 
            WHERE symbol = '{symbol}' 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        result = client.query(query)

        if not result.result_rows:
            return None

        row = result.result_rows[0]
        return {
            "symbol": symbol,
            "ar_coeff": row[0],
            "ma_coeff": row[1],
            "garch_volatility": row[2],
            "calculated_at": row[3],
        }
    except Exception as e:
        logger.error("Error fetching metrics for %s: %s", symbol, e)
        return None
