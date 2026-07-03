import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_FILE = DATA_DIR / "weather_data.csv"
KAGGLE_DATA_FILE = DATA_DIR / "GlobalWeatherRepository.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib_cache"

OUTPUT_DIR.mkdir(exist_ok=True)
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt


COLUMN_ALIASES = {
    "Date": ["Date", "date", "last_updated", "Last Updated"],
    "Country": ["Country", "country"],
    "City": ["City", "city", "location_name", "Location", "location"],
    "Temperature_C": ["Temperature_C", "temperature_celsius", "temp_c", "Temperature"],
    "Humidity_Percent": ["Humidity_Percent", "humidity", "Humidity"],
    "Rainfall_mm": ["Rainfall_mm", "precip_mm", "precipitation_mm", "Rainfall"],
    "WindSpeed_kmph": ["WindSpeed_kmph", "wind_kph", "WindSpeed", "wind_speed_kmph"],
}


def find_data_file(requested_path: str | None) -> Path:
    if requested_path:
        return Path(requested_path).expanduser().resolve()
    if KAGGLE_DATA_FILE.exists():
        return KAGGLE_DATA_FILE
    return DEFAULT_DATA_FILE


def pick_column(df: pd.DataFrame, logical_name: str) -> str | None:
    lower_lookup = {column.lower(): column for column in df.columns}
    for candidate in COLUMN_ALIASES[logical_name]:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    return None


def load_weather_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    normalized = pd.DataFrame()

    required = ["Date", "City", "Temperature_C", "Humidity_Percent", "Rainfall_mm", "WindSpeed_kmph"]
    missing = []
    for logical_name in required:
        source_column = pick_column(raw, logical_name)
        if source_column is None:
            missing.append(logical_name)
        else:
            normalized[logical_name] = raw[source_column]

    country_column = pick_column(raw, "Country")
    normalized["Country"] = raw[country_column] if country_column else "Unknown"

    if missing:
        raise ValueError(
            "Missing required weather columns: "
            + ", ".join(missing)
            + "\nUse the Kaggle World Weather Repository CSV or match the sample CSV format."
        )

    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    numeric_columns = ["Temperature_C", "Humidity_Percent", "Rainfall_mm", "WindSpeed_kmph"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized["Country"] = normalized["Country"].fillna("Unknown").astype(str)
    normalized["City"] = normalized["City"].fillna("Unknown").astype(str)
    normalized = normalized.dropna(subset=["Date", *numeric_columns]).sort_values("Date")
    normalized["Month"] = normalized["Date"].dt.month_name()
    normalized["Month_Number"] = normalized["Date"].dt.month
    normalized = add_weather_markers(normalized)
    return normalized


def classify_weather(row: pd.Series) -> tuple[str, str]:
    if (
        row["Temperature_C"] >= 40
        or row["Temperature_C"] <= 0
        or row["Rainfall_mm"] >= 80
        or row["WindSpeed_kmph"] >= 60
    ):
        return "Extreme", "Red"
    if (
        row["Temperature_C"] >= 35
        or row["Temperature_C"] <= 5
        or row["Rainfall_mm"] >= 25
        or row["WindSpeed_kmph"] >= 35
        or row["Humidity_Percent"] >= 85
    ):
        return "Warning", "Yellow"
    return "Normal", "Green"


def add_weather_markers(df: pd.DataFrame) -> pd.DataFrame:
    marked = df.copy()
    markers = marked.apply(classify_weather, axis=1, result_type="expand")
    marked["Weather_Status"] = markers[0]
    marked["Marker_Color"] = markers[1]
    return marked


def filter_weather_data(df: pd.DataFrame, country: str | None, city: str | None) -> pd.DataFrame:
    filtered = df.copy()
    if country:
        filtered = filtered[filtered["Country"].str.casefold() == country.casefold()]
    if city:
        filtered = filtered[filtered["City"].str.casefold() == city.casefold()]
    if filtered.empty:
        raise ValueError("No records found for the selected country/city filter.")
    return filtered


def analyze_weather(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = (
        df.groupby(["Month_Number", "Month"], as_index=False)
        .agg(
            Average_Temperature_C=("Temperature_C", "mean"),
            Total_Rainfall_mm=("Rainfall_mm", "sum"),
            Average_Humidity_Percent=("Humidity_Percent", "mean"),
            Average_WindSpeed_kmph=("WindSpeed_kmph", "mean"),
        )
        .sort_values("Month_Number")
    )

    country_summary = (
        df.groupby("Country", as_index=False)
        .agg(
            Records=("Date", "count"),
            Average_Temperature_C=("Temperature_C", "mean"),
            Total_Rainfall_mm=("Rainfall_mm", "sum"),
            Average_Humidity_Percent=("Humidity_Percent", "mean"),
            Average_WindSpeed_kmph=("WindSpeed_kmph", "mean"),
        )
        .sort_values("Average_Temperature_C", ascending=False)
    )

    city_summary = (
        df.groupby(["Country", "City"], as_index=False)
        .agg(
            Records=("Date", "count"),
            Average_Temperature_C=("Temperature_C", "mean"),
            Total_Rainfall_mm=("Rainfall_mm", "sum"),
            Extreme_Days=("Weather_Status", lambda values: (values == "Extreme").sum()),
            Warning_Days=("Weather_Status", lambda values: (values == "Warning").sum()),
            Normal_Days=("Weather_Status", lambda values: (values == "Normal").sum()),
        )
        .sort_values(["Country", "City"])
    )

    city_trends = calculate_city_trends(df)
    city_summary = city_summary.merge(city_trends, on=["Country", "City"], how="left")

    hottest_day = df.loc[df["Temperature_C"].idxmax()]
    coldest_day = df.loc[df["Temperature_C"].idxmin()]
    rainiest_day = df.loc[df["Rainfall_mm"].idxmax()]
    windiest_day = df.loc[df["WindSpeed_kmph"].idxmax()]

    trend = np.polyfit(np.arange(len(df)), df["Temperature_C"], 1)
    predicted_temperatures = predict_next_temperatures(df, days=7)
    city_predictions = predict_city_temperatures(df, days=7)

    insights = {
        "records": len(df),
        "countries": df["Country"].nunique(),
        "cities": df["City"].nunique(),
        "hottest_day": hottest_day,
        "coldest_day": coldest_day,
        "rainiest_day": rainiest_day,
        "windiest_day": windiest_day,
        "avg_temperature": df["Temperature_C"].mean(),
        "avg_humidity": df["Humidity_Percent"].mean(),
        "total_rainfall": df["Rainfall_mm"].sum(),
        "avg_wind_speed": df["WindSpeed_kmph"].mean(),
        "temperature_trend_slope": trend[0],
        "rainiest_month": monthly.loc[monthly["Total_Rainfall_mm"].idxmax()],
        "hottest_month": monthly.loc[monthly["Average_Temperature_C"].idxmax()],
        "coldest_month": monthly.loc[monthly["Average_Temperature_C"].idxmin()],
    }

    return insights, monthly, country_summary, city_summary, predicted_temperatures, city_predictions


def predict_next_temperatures(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    daily = df.groupby("Date", as_index=False)["Temperature_C"].mean().sort_values("Date")
    x = np.arange(len(daily))
    y = daily["Temperature_C"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(daily), len(daily) + days)
    future_dates = pd.date_range(daily["Date"].max() + pd.Timedelta(days=1), periods=days)
    predictions = slope * future_x + intercept

    return pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted_Temperature_C": np.round(predictions, 2),
        }
    )


def trend_label(slope: float) -> str:
    if slope > 0.03:
        return "Increasing"
    if slope < -0.03:
        return "Decreasing"
    return "Stable"


def calculate_city_trends(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, city), group in df.groupby(["Country", "City"]):
        daily = group.groupby("Date", as_index=False)["Temperature_C"].mean().sort_values("Date")
        if len(daily) < 2:
            slope = 0.0
        else:
            slope = float(np.polyfit(np.arange(len(daily)), daily["Temperature_C"], 1)[0])
        rows.append(
            {
                "Country": country,
                "City": city,
                "Temperature_Trend_Slope": round(slope, 4),
                "Future_Trend": trend_label(slope),
            }
        )
    return pd.DataFrame(rows)


def predict_city_temperatures(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    rows = []
    for (country, city), group in df.groupby(["Country", "City"]):
        daily = group.groupby("Date", as_index=False)["Temperature_C"].mean().sort_values("Date")
        if len(daily) < 2:
            slope = 0.0
            intercept = float(daily["Temperature_C"].iloc[-1])
        else:
            slope, intercept = np.polyfit(np.arange(len(daily)), daily["Temperature_C"], 1)
        future_dates = pd.date_range(daily["Date"].max() + pd.Timedelta(days=1), periods=days)
        for offset, future_date in enumerate(future_dates, start=len(daily)):
            rows.append(
                {
                    "Country": country,
                    "City": city,
                    "Date": future_date.date(),
                    "Predicted_Temperature_C": round(float(slope * offset + intercept), 2),
                    "Future_Trend": trend_label(float(slope)),
                }
            )
    return pd.DataFrame(rows)


def place_label(row: pd.Series) -> str:
    country = row.get("Country", "Unknown")
    city = row.get("City", "Unknown")
    return f"{city}, {country}" if country != "Unknown" else city


def create_charts(
    df: pd.DataFrame,
    monthly: pd.DataFrame,
    country_summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    title_suffix: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Weather Data Analysis Dashboard - {title_suffix}", fontsize=18, fontweight="bold")

    daily = df.groupby("Date", as_index=False)["Temperature_C"].mean().sort_values("Date")
    axes[0, 0].plot(daily["Date"], daily["Temperature_C"], color="#e63946", linewidth=2.3)
    axes[0, 0].set_title("Temperature Trend", fontweight="bold")
    axes[0, 0].set_ylabel("Temperature (C)")
    axes[0, 0].tick_params(axis="x", rotation=35)

    axes[0, 1].bar(monthly["Month"].str[:3], monthly["Total_Rainfall_mm"], color="#1d70a2")
    axes[0, 1].set_title("Monthly Rainfall", fontweight="bold")
    axes[0, 1].set_ylabel("Rainfall (mm)")

    axes[1, 0].plot(monthly["Month"].str[:3], monthly["Average_Temperature_C"], color="#ff9f1c", marker="D", linewidth=2.5)
    axes[1, 0].set_title("Average Monthly Temperature", fontweight="bold")
    axes[1, 0].set_ylabel("Average Temperature (C)")

    scatter = axes[1, 1].scatter(
        df["Humidity_Percent"],
        df["Temperature_C"],
        c=df["Marker_Color"].map({"Green": "#2a9d8f", "Yellow": "#f4a261", "Red": "#e63946"}),
        s=70,
        edgecolors="#222222",
        alpha=0.8,
    )
    axes[1, 1].set_title("Humidity vs Temperature", fontweight="bold")
    axes[1, 1].set_xlabel("Humidity (%)")
    axes[1, 1].set_ylabel("Temperature (C)")
    legend_items = [
        plt.Line2D([0], [0], marker="o", color="w", label="Normal - Green", markerfacecolor="#2a9d8f", markersize=9),
        plt.Line2D([0], [0], marker="o", color="w", label="Warning - Yellow", markerfacecolor="#f4a261", markersize=9),
        plt.Line2D([0], [0], marker="o", color="w", label="Extreme - Red", markerfacecolor="#e63946", markersize=9),
    ]
    axes[1, 1].legend(handles=legend_items, loc="best")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "weather_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(daily["Date"], daily["Temperature_C"], label="Actual Average Temperature", color="#264653", linewidth=2.4)
    plt.plot(
        predictions["Date"],
        predictions["Predicted_Temperature_C"],
        label="Predicted Temperature",
        color="#e76f51",
        linestyle="--",
        marker="o",
        linewidth=2.4,
    )
    plt.title("7-Day Future Temperature Prediction", fontsize=16, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "temperature_prediction.png", dpi=180, bbox_inches="tight")
    plt.close()

    top_countries = country_summary.head(10).sort_values("Average_Temperature_C")
    plt.figure(figsize=(12, 6))
    plt.barh(top_countries["Country"], top_countries["Average_Temperature_C"], color="#2a9d8f")
    plt.title("Top 10 Countries by Average Temperature", fontsize=16, fontweight="bold")
    plt.xlabel("Average Temperature (C)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_countries_temperature.png", dpi=180, bbox_inches="tight")
    plt.close()

    trend_colors = {"Increasing": "#e63946", "Stable": "#2a9d8f", "Decreasing": "#457b9d"}
    top_cities = city_summary.sort_values("Records", ascending=False).head(20)
    plt.figure(figsize=(13, 7))
    plt.barh(
        top_cities["City"] + ", " + top_cities["Country"],
        top_cities["Temperature_Trend_Slope"],
        color=top_cities["Future_Trend"].map(trend_colors),
    )
    plt.title("Future Temperature Trend by City", fontsize=16, fontweight="bold")
    plt.xlabel("Temperature Trend Slope")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "city_future_trends.png", dpi=180, bbox_inches="tight")
    plt.close()


def write_report(
    insights: dict,
    monthly: pd.DataFrame,
    country_summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    city_predictions: pd.DataFrame,
    source_file: Path,
    title_suffix: str,
) -> None:
    hottest = insights["hottest_day"]
    coldest = insights["coldest_day"]
    rainiest = insights["rainiest_day"]
    windiest = insights["windiest_day"]
    trend_text = "increasing" if insights["temperature_trend_slope"] > 0 else "decreasing"

    report = f"""
WEATHER DATA ANALYSIS SYSTEM
Dataset Source: {source_file.name}
Analysis Scope: {title_suffix}
Total Records Analyzed: {insights["records"]}
Countries Covered: {insights["countries"]}
Cities Covered: {insights["cities"]}

KEY INSIGHTS
1. Hottest Record: {hottest["Date"].date()} at {place_label(hottest)} with {hottest["Temperature_C"]:.1f} C
2. Coldest Record: {coldest["Date"].date()} at {place_label(coldest)} with {coldest["Temperature_C"]:.1f} C
3. Rainiest Record: {rainiest["Date"].date()} at {place_label(rainiest)} with {rainiest["Rainfall_mm"]:.1f} mm rainfall
4. Windiest Record: {windiest["Date"].date()} at {place_label(windiest)} with {windiest["WindSpeed_kmph"]:.1f} kmph wind speed
5. Average Temperature: {insights["avg_temperature"]:.2f} C
6. Average Humidity: {insights["avg_humidity"]:.2f}%
7. Total Rainfall: {insights["total_rainfall"]:.2f} mm
8. Average Wind Speed: {insights["avg_wind_speed"]:.2f} kmph
9. Temperature Trend: Overall {trend_text}

MONTHLY HIGHLIGHTS
Hottest Month: {insights["hottest_month"]["Month"]} ({insights["hottest_month"]["Average_Temperature_C"]:.2f} C average)
Coldest Month: {insights["coldest_month"]["Month"]} ({insights["coldest_month"]["Average_Temperature_C"]:.2f} C average)
Rainiest Month: {insights["rainiest_month"]["Month"]} ({insights["rainiest_month"]["Total_Rainfall_mm"]:.2f} mm rainfall)

TOP COUNTRIES BY AVERAGE TEMPERATURE
{country_summary[["Country", "Records", "Average_Temperature_C", "Total_Rainfall_mm"]].head(10).round(2).to_string(index=False)}

CITY-WISE FUTURE TREND
{city_summary[["Country", "City", "Records", "Average_Temperature_C", "Extreme_Days", "Warning_Days", "Normal_Days", "Future_Trend"]].head(30).round(2).to_string(index=False)}

AVERAGE MONTHLY WEATHER
{monthly[["Month", "Average_Temperature_C", "Total_Rainfall_mm", "Average_Humidity_Percent", "Average_WindSpeed_kmph"]].round(2).to_string(index=False)}

7-DAY WORLD AVERAGE TEMPERATURE PREDICTION
{predictions.to_string(index=False)}

7-DAY CITY-WISE TEMPERATURE PREDICTION
{city_predictions.head(80).to_string(index=False)}
""".strip()

    (OUTPUT_DIR / "analysis_report.txt").write_text(report, encoding="utf-8")


def print_console_summary(insights: dict, title_suffix: str, source_file: Path) -> None:
    hottest = insights["hottest_day"]
    coldest = insights["coldest_day"]
    rainiest_month = insights["rainiest_month"]

    print("\nWeather Data Analysis System")
    print("-" * 38)
    print(f"Dataset: {source_file.name}")
    print(f"Scope: {title_suffix}")
    print(f"Records analyzed: {insights['records']}")
    print(f"Countries covered: {insights['countries']}")
    print(f"Cities covered: {insights['cities']}")
    print(f"Hottest record: {place_label(hottest)} on {hottest['Date'].date()} ({hottest['Temperature_C']:.1f} C)")
    print(f"Coldest record: {place_label(coldest)} on {coldest['Date'].date()} ({coldest['Temperature_C']:.1f} C)")
    print(f"Rainiest month: {rainiest_month['Month']} ({rainiest_month['Total_Rainfall_mm']:.1f} mm)")
    print("Charts saved in: output/")
    print("Report saved as: output/analysis_report.txt\n")


def build_title_suffix(country: str | None, city: str | None) -> str:
    if country and city:
        return f"{city}, {country}"
    if country:
        return country
    if city:
        return city
    return "Worldwide"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze worldwide weather data from CSV/Kaggle.")
    parser.add_argument("--data", help="Path to Kaggle or sample weather CSV file.")
    parser.add_argument("--country", help="Optional country filter, for example India.")
    parser.add_argument("--city", help="Optional city filter, for example Chandigarh.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_file = find_data_file(args.data)
    df = load_weather_data(source_file)
    df = filter_weather_data(df, args.country, args.city)
    title_suffix = build_title_suffix(args.country, args.city)

    insights, monthly, country_summary, city_summary, predictions, city_predictions = analyze_weather(df)
    df.to_csv(OUTPUT_DIR / "marked_weather_records.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "monthly_summary.csv", index=False)
    country_summary.to_csv(OUTPUT_DIR / "country_summary.csv", index=False)
    city_summary.to_csv(OUTPUT_DIR / "city_trend_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "temperature_predictions.csv", index=False)
    city_predictions.to_csv(OUTPUT_DIR / "city_temperature_predictions.csv", index=False)
    create_charts(df, monthly, country_summary, city_summary, predictions, title_suffix)
    write_report(insights, monthly, country_summary, city_summary, predictions, city_predictions, source_file, title_suffix)
    print_console_summary(insights, title_suffix, source_file)


if __name__ == "__main__":
    main()
