import numpy as np
import pandas as pd


def generate(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n)
    gender = rng.choice(["Male", "Female"], n)
    income = rng.normal(60000, 20000, n).clip(15000, 150000).round(0)
    browsing_time = rng.gamma(3, 6, n).clip(1, 90).round(1)          # minutes
    pages_visited = rng.poisson(8, n).clip(1, 40)
    previous_purchases = rng.poisson(2, n).clip(0, 15)
    time_on_site = (browsing_time * rng.uniform(0.8, 1.6, n)).round(1)  # minutes
    discount_offered = rng.choice([0, 5, 10, 15, 20, 25, 30], n)

    z = (-7.0
         + 0.07 * browsing_time
         + 0.2 * pages_visited
         + 0.6 * previous_purchases
         + 0.00002 * income
         + 0.12 * discount_offered
         + 0.01 * time_on_site
         - 0.015 * age
         + rng.normal(0, 0.5, n))
    purchase = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "annual_income": income,
        "browsing_time": browsing_time,
        "pages_visited": pages_visited,
        "previous_purchases": previous_purchases,
        "time_on_site": time_on_site,
        "discount_offered": discount_offered,
        "purchase": purchase,
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/customer_data.csv", index=False)
    print(df.shape, "purchase rate:", round(df["purchase"].mean(), 3))
