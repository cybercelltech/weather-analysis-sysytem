from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from weather_analysis import (
    DEFAULT_DATA_FILE,
    KAGGLE_DATA_FILE,
    analyze_weather,
    filter_weather_data,
    find_data_file,
    load_weather_data,
)
STATUS_COLORS = {
    "Normal": "#2a9d8f",
    "Warning": "#f4a261",
    "Extreme": "#e63946",
}


st.set_page_config(
    page_title="Weather Data Analysis System",
    page_icon="W",
    layout="wide",
)


@st.cache_data
def cached_load_data(csv_path: str) -> pd.DataFrame:
    return load_weather_data(Path(csv_path))


def source_file_label() -> str:
    if KAGGLE_DATA_FILE.exists():
        return f"Kaggle dataset: {KAGGLE_DATA_FILE.name}"
    return f"Demo dataset: {DEFAULT_DATA_FILE.name}"


def make_temperature_chart(df: pd.DataFrame) -> plt.Figure:
    daily = df.groupby("Date", as_index=False)["Temperature_C"].mean().sort_values("Date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["Date"], daily["Temperature_C"], color="#e63946", linewidth=2.4)
    ax.set_title("Temperature Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (C)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return fig


def make_rainfall_chart(monthly: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly["Month"].str[:3], monthly["Total_Rainfall_mm"], color="#1d70a2")
    ax.set_title("Monthly Rainfall")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rainfall (mm)")
    ax.grid(axis="y", alpha=0.25)
    return fig


def make_status_chart(df: pd.DataFrame) -> plt.Figure:
    counts = df["Weather_Status"].value_counts().reindex(["Normal", "Warning", "Extreme"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color=[STATUS_COLORS[label] for label in counts.index])
    ax.set_title("Weather Status Marking")
    ax.set_xlabel("Status")
    ax.set_ylabel("Number of Records")
    ax.grid(axis="y", alpha=0.25)
    return fig


def make_prediction_chart(predictions: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        predictions["Date"],
        predictions["Predicted_Temperature_C"],
        color="#e76f51",
        linestyle="--",
        marker="o",
        linewidth=2.4,
    )
    ax.set_title("7-Day Future Temperature Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (C)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return fig


st.title("Weather Data Analysis System")
st.caption("Worldwide city weather analysis with red, yellow, green marking and future trend prediction.")

source_file = find_data_file(None)
df = cached_load_data(str(source_file))

with st.sidebar:
    st.header("Filters")
    st.write(source_file_label())

    countries = ["All"] + sorted(df["Country"].dropna().unique().tolist())
    selected_country = st.selectbox("Country", countries)

    country_filter = None if selected_country == "All" else selected_country
    city_options_df = filter_weather_data(df, country_filter, None) if country_filter else df
    cities = ["All"] + sorted(city_options_df["City"].dropna().unique().tolist())
    selected_city = st.selectbox("City", cities)
    city_filter = None if selected_city == "All" else selected_city

    st.divider()
    st.write("Marker meaning")
    st.markdown(":green[Green] = Normal")
    st.markdown(":orange[Yellow] = Warning")
    st.markdown(":red[Red] = Extreme")

filtered_df = filter_weather_data(df, country_filter, city_filter)
insights, monthly, country_summary, city_summary, predictions, city_predictions = analyze_weather(filtered_df)

metric_cols = st.columns(5)
metric_cols[0].metric("Records", f"{insights['records']:,}")
metric_cols[1].metric("Countries", insights["countries"])
metric_cols[2].metric("Cities", insights["cities"])
metric_cols[3].metric("Avg Temp", f"{insights['avg_temperature']:.1f} C")
metric_cols[4].metric("Total Rain", f"{insights['total_rainfall']:.0f} mm")

st.subheader("Extreme Weather Highlights")
highlight_cols = st.columns(4)
hottest = insights["hottest_day"]
coldest = insights["coldest_day"]
rainiest = insights["rainiest_day"]
windiest = insights["windiest_day"]
highlight_cols[0].metric("Hottest", f"{hottest['Temperature_C']:.1f} C", f"{hottest['City']}, {hottest['Country']}")
highlight_cols[1].metric("Coldest", f"{coldest['Temperature_C']:.1f} C", f"{coldest['City']}, {coldest['Country']}")
highlight_cols[2].metric("Rainiest", f"{rainiest['Rainfall_mm']:.1f} mm", f"{rainiest['City']}, {rainiest['Country']}")
highlight_cols[3].metric("Windiest", f"{windiest['WindSpeed_kmph']:.1f} kmph", f"{windiest['City']}, {windiest['Country']}")

chart_cols = st.columns(2)
chart_cols[0].pyplot(make_temperature_chart(filtered_df), clear_figure=True)
chart_cols[1].pyplot(make_rainfall_chart(monthly), clear_figure=True)

chart_cols = st.columns(2)
chart_cols[0].pyplot(make_status_chart(filtered_df), clear_figure=True)
chart_cols[1].pyplot(make_prediction_chart(predictions), clear_figure=True)

st.subheader("City Future Trend Prediction")
st.dataframe(
    city_summary[
        [
            "Country",
            "City",
            "Records",
            "Average_Temperature_C",
            "Extreme_Days",
            "Warning_Days",
            "Normal_Days",
            "Future_Trend",
        ]
    ].round(2),
    use_container_width=True,
)

st.subheader("7-Day City-Wise Temperature Prediction")
st.dataframe(city_predictions, use_container_width=True)

st.subheader("Marked Weather Records")
st.dataframe(
    filtered_df[
        [
            "Date",
            "Country",
            "City",
            "Temperature_C",
            "Humidity_Percent",
            "Rainfall_mm",
            "WindSpeed_kmph",
            "Weather_Status",
            "Marker_Color",
        ]
    ],
    use_container_width=True,
)

st.download_button(
    "Download city predictions CSV",
    city_predictions.to_csv(index=False),
    file_name="city_temperature_predictions.csv",
    mime="text/csv",
)



