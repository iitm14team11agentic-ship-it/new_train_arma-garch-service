import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")


def train_and_extract_params(prices):
    """
    Input: A list or array of historical Close prices (floats).
    Output: A dictionary containing trained parameters for ARMA and GARCH.
    """

    # returns = 100 * ln(Price_t / Price_t-1)
    price_series = pd.Series(prices)
    returns = 100 * np.log(price_series / price_series.shift(1)).dropna()

    # We'll return a standardized flattened dict of metrics.
    # Keys: success, ar_coeff, ma_coeff, const, omega, alpha, beta, garch_volatility
    results = {
        "success": True,
        "ar_coeff": 0.0,
        "ma_coeff": 0.0,
        "const": 0.0,
        "omega": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "garch_volatility": 0.0,
    }

    try:
        # Validate input length — ARIMA/GARCH require more than a few observations
        if len(returns) < 10:
            return {
                "error": "Not enough return observations to train models",
                "success": False,
            }
        # ARMA(1,1) model
        # order=(p, d, q) -> (1, 0, 1)
        arma_model = ARIMA(returns, order=(1, 0, 1))
        arma_fit = arma_model.fit()

        # Populate flattened ARMA output
        results["ar_coeff"] = float(arma_fit.params.get("ar.L1", 0))
        results["ma_coeff"] = float(arma_fit.params.get("ma.L1", 0))
        results["const"] = float(arma_fit.params.get("const", 0))

        # GARCH(1,1)
        garch = arch_model(returns, vol="Garch", p=1, q=1, dist="Normal")
        garch_fit = garch.fit(disp="off")
        # disp='off' hides training logs

        # Populate flattened GARCH output
        # Use .get to avoid KeyError if param names differ across versions
        results["omega"] = float(
            garch_fit.params.get("omega", 0.0)
        )  # Baseline variance
        results["alpha"] = float(
            garch_fit.params.get("alpha[1]", garch_fit.params.get("alpha", 0.0))
        )
        results["beta"] = float(
            garch_fit.params.get("beta[1]", garch_fit.params.get("beta", 0.0))
        )
        # Most recent volatility estimate (sigma^2)
        results["garch_volatility"] = float(garch_fit.conditional_volatility.iloc[-1])

        return results

    except Exception as e:
        return {"error": str(e), "success": False}
