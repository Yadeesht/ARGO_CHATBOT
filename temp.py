from app import parse_bgc_tool_input
from app import parse_action_input
import pandas as pd
import os
import json
import re
import traceback
import sys
from datetime import datetime 
from argopy import ArgoIndex
import matplotlib.pyplot as plt
import plotly.express as px

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
from langchain.prompts import PromptTemplate

from argopy import DataFetcher

def fetch_argo_data_by_region_plot(tool_input):
    """
    Flexible ARGO data fetching tool that can handle:
    - Single coordinates (lat, lon) or bounding box (lat_min, lat_max, lon_min, lon_max)
    - Any ARGO parameters as filters: CYCLE_NUMBER, DATA_MODE, PLATFORM_NUMBER, etc.
    - Optional depth and time constraints
    - Parameter-based searches (TEMP, PSAL, PRES, etc.)
    
    Examples:
    {"lat": 15.5, "lon": 88.2} - Single point
    {"lat_min": 10, "lat_max": 20, "lon_min": 80, "lon_max": 90} - Bounding box
    {"platform_number": "2903334"} - Specific float
    {"temp_min": 20, "temp_max": 30} - Temperature range
    {"plot":True} or {"plot":False} - Whether to generate a plot if user asked or if possible for the given query then suggested
    {"plot_opt": ["scatter","scatter_3d","scatter_polar","scatter_ternary","line","line_3d","line_polar","area","bar","histogram","violin","box","strip","pie","sunburst","treemap","icicle","funnel","funnel_area","density_contour","density_heatmap","scatter_geo","choropleth","choropleth_mapbox","scatter_mapbox","density_mapbox","parallel_coordinates","parallel_categories","imshow"]} - Plotting options for best visualization 

    {"x": "PRES", "y": "TEMP"} - Plotting parameters if plot is True
    whenever plot is True plot_opt must be provided and x and y must be provided
    In plot_opt the minimum number of plots is 3 regardless of the user prompt,for the data fetched the best three plots will be generated.
    Make sure to provide valid parameters for x and y that exist in the fetched data.
    Make sure to include only those plot options int plot_opt which are neccesary for the data fetched and are valid for the data fetched. We dont want to maximize the number of plots but we want to provide the best possible plots for the data fetched.
    If two plot options visualize the data in similar ways then include only one of them.
    We need to make sure to optimize the number of plots and the quality of plots.Such that the user is not overwhelmed with too many plots and the plots provided are of high quality and provide good insights about the data fetched.
    If there is a timeout error while fetching the data,then first prioritize decreasing the date range to fetch the data to a smaller range and then if the error still persists then prioritize decreasing the bounding box to a smaller box.
    Returns a image path.
    """
    try:
        parsed = parse_action_input(tool_input)
        print(f"Parsed input: {parsed}")
        
        argo_params = [
            "CYCLE_NUMBER", "DATA_MODE", "DIRECTION", "PLATFORM_NUMBER",
            "POSITION_QC", "PRES", "PRES_ERROR", "PRES_QC",
            "PSAL", "PSAL_ERROR", "PSAL_QC", 
            "TEMP", "TEMP_ERROR", "TEMP_QC",
            "TIME_QC", "LATITUDE", "LONGITUDE", "TIME"
        ]
        
        query_params = {}

        if "lat" in parsed and "lon" in parsed:
            # Single point - create small bounding box around it
            lat = float(parsed["lat"])
            lon = float(parsed["lon"])
            buffer = float(parsed.get("buffer", 1.0))  # Default 1 degree buffer
            query_params.update({
                "lat_min": lat - buffer,
                "lat_max": lat + buffer,
                "lon_min": lon - buffer,
                "lon_max": lon + buffer
            })
            print(f"Single point query: {lat}, {lon} with {buffer}° buffer")
            
        elif all(k in parsed for k in ["lat_min", "lat_max", "lon_min", "lon_max"]):
            # Full bounding box
            query_params.update({
                "lat_min": float(parsed["lat_min"]),
                "lat_max": float(parsed["lat_max"]),
                "lon_min": float(parsed["lon_min"]),
                "lon_max": float(parsed["lon_max"])
            })
            print("Bounding box query")
            
        elif any(k in parsed for k in ["lat_min", "lat_max", "lon_min", "lon_max"]):
            # Partial coordinates - fill defaults
            query_params.update({
                "lat_min": float(parsed.get("lat_min", -90)),
                "lat_max": float(parsed.get("lat_max", 90)),
                "lon_min": float(parsed.get("lon_min", -180)),
                "lon_max": float(parsed.get("lon_max", 180))
            })
            print("Partial coordinates - using global defaults for missing bounds")
        else:
            # No geographic constraints - global search
            query_params.update({
                "lat_min": -90, "lat_max": 90,
                "lon_min": -180, "lon_max": 180
            })
            print("Global search - no geographic constraints")

        query_params["pres_min"] = float(parsed.get("pres_min", 0))
        query_params["pres_max"] = float(parsed.get("pres_max", 2000))

        query_params["date_start"] = parsed.get("date_start", "2020-01-01")
        query_params["date_end"] = parsed.get("date_end", "2024-12-31")

        box_numeric = [
            query_params["lon_min"], query_params["lon_max"],
            query_params["lat_min"], query_params["lat_max"],
            query_params["pres_min"], query_params["pres_max"],
            query_params["date_start"], query_params["date_end"]
        ]

        ds = DataFetcher(fs_opts={'timeout': 150}).region(box_numeric).to_xarray()
        df = ds.to_dataframe()

        if df.empty:
            return "No data found for the specified region and time."

        # Reset index (N_POINTS can be an index) and summarise
        df_reset = df.reset_index()
        # some datasets may not have these columns - guard with get()
        if "platform_number" in parsed:
            platform = str(parsed["platform_number"])
            if "PLATFORM_NUMBER" in df_reset.columns:
                df_reset = df_reset[df_reset["PLATFORM_NUMBER"].astype(str) == platform]
                filters_applied.append(f"Platform {platform}")
        
        # Filter by cycle number
        if "cycle_number" in parsed:
            cycle = int(parsed["cycle_number"])
            if "CYCLE_NUMBER" in df_reset.columns:
                df_reset = df_reset[df_reset["CYCLE_NUMBER"] == cycle]
                filters_applied.append(f"Cycle {cycle}")
        
        # Filter by temperature range
        if "temp_min" in parsed or "temp_max" in parsed:
            if "TEMP" in df_reset.columns:
                temp_mask = pd.Series([True] * len(df_reset))
                if "temp_min" in parsed:
                    temp_min = float(parsed["temp_min"])
                    temp_mask &= (df_reset["TEMP"] >= temp_min)
                    filters_applied.append(f"Temp ≥ {temp_min}°C")
                if "temp_max" in parsed:
                    temp_max = float(parsed["temp_max"])
                    temp_mask &= (df_reset["TEMP"] <= temp_max)
                    filters_applied.append(f"Temp ≤ {temp_max}°C")
                df_reset = df_reset[temp_mask]
        
        # Filter by salinity range
        if "psal_min" in parsed or "psal_max" in parsed:
            if "PSAL" in df_reset.columns:
                psal_mask = pd.Series([True] * len(df_reset))
                if "psal_min" in parsed:
                    psal_min = float(parsed["psal_min"])
                    psal_mask &= (df_reset["PSAL"] >= psal_min)
                    filters_applied.append(f"Salinity ≥ {psal_min}")
                if "psal_max" in parsed:
                    psal_max = float(parsed["psal_max"])
                    psal_mask &= (df_reset["PSAL"] <= psal_max)
                    filters_applied.append(f"Salinity ≤ {psal_max}")
                df_reset = df_reset[psal_mask]
        
        # Filter by pressure/depth range
        if "pres_min" in parsed or "pres_max" in parsed:
            if "PRES" in df_reset.columns:
                pres_mask = pd.Series([True] * len(df_reset))
                if "pres_min" in parsed:
                    pres_min = float(parsed["pres_min"])
                    pres_mask &= (df_reset["PRES"] >= pres_min)
                    filters_applied.append(f"Pressure ≥ {pres_min} dbar")
                if "pres_max" in parsed:
                    pres_max = float(parsed["pres_max"])
                    pres_mask &= (df_reset["PRES"] <= pres_max)
                    filters_applied.append(f"Pressure ≤ {pres_max} dbar")
                df_reset = df_reset[pres_mask]
        
        # Filter by data mode
        if "data_mode" in parsed:
            data_mode = parsed["data_mode"].upper()
            if "DATA_MODE" in df_reset.columns:
                df_reset = df_reset[df_reset["DATA_MODE"] == data_mode]
                filters_applied.append(f"Data mode: {data_mode}")

        final_count = len(df_reset)

        if "TIME" in df_reset.columns:
            min_date = pd.to_datetime(df_reset["TIME"].min()).strftime("%Y-%m-%d")
            max_date = pd.to_datetime(df_reset["TIME"].max()).strftime("%Y-%m-%d")
            date_info = f"between {min_date} and {max_date}"
        else:
            date_info = f"in requested period {query_params['date_start']} to {query_params['date_end']}"

        # Parameter ranges
        param_stats = []
        if "TEMP" in df_reset.columns:
            temp_range = f"{df_reset['TEMP'].min():.2f} to {df_reset['TEMP'].max():.2f}°C"
            param_stats.append(f"🌡️ Temperature: {temp_range}")
        
        if "PSAL" in df_reset.columns:
            psal_range = f"{df_reset['PSAL'].min():.2f} to {df_reset['PSAL'].max():.2f}"
            param_stats.append(f"🧂 Salinity: {psal_range}")
            
        if "PRES" in df_reset.columns:
            pres_range = f"{df_reset['PRES'].min():.1f} to {df_reset['PRES'].max():.1f} dbar"
            param_stats.append(f"📊 Pressure: {pres_range}")
        
        if parsed["plot"]==True:
            if parsed["plot_opt"] == "scatter":
                fig = px.scatter(data_frame=df_reset, x=parsed["x"], y=parsed["y"],title=f"Agro Data: {parsed['x']} vs. {parsed['y']}")
            elif parsed["plot_opt"] == "line":
                fig = px.line(data_frame=df_reset, x=parsed["x"], y=parsed["y"],title=f"Agro Data: {parsed['x']} vs. {parsed['y']}")
            else:
                return "❌ Invalid kind. Choose 'scatter' or 'line'."

            os.makedirs("out_img", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plot_{parsed['plot_opt']}_{parsed['x']}_{parsed['y']}_{timestamp}.png"
            image_path = os.path.join("out_img", filename)
            fig.write_image(image_path)

            return f"✅ Plot generated and saved to {image_path}"
    except ValueError as ve:
        return f"Input parsing error: {ve}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error fetching data: {e}\nTraceback:\n{tb}"

def generate_bgc_parameter_map(tool_input):
    """
    Create a global scatter map of data-quality modes for a given
    Bio-Geo-Chemical (BGC) parameter from the Argo float dataset.

    • Expects: JSON with a key "parameter" (e.g. {"parameter": "DOXY"})
    • Finds all float profiles containing that parameter.
    • Converts their data-mode flags (R/A/D) to numeric values.
    • Plots a world map colored by data-mode and saves it as a PNG.

    Useful for questions like:
    – "Show where dissolved oxygen data exists and its quality mode."
    – "Map nitrate observations and highlight validated profiles."
    """
    try:
        parameter = parse_bgc_tool_input(tool_input)
        if not parameter:
            raise ValueError("Input is missing the 'parameter' field.")

        idx = ArgoIndex(index_file='bgc-b').load()

        x=idx.read_params(parameter)

        if not x:
            return f"No data found for the BGC parameter: {parameter}"
        
        df = idx.to_dataframe() 

        df["variables"] = df["parameters"].apply(lambda x: x.split())
        df[f"{parameter}_DM"] = df.apply(lambda x: x['parameter_data_mode'][x['variables'].index(parameter)] if parameter in x['variables'] else '', axis=1)

        df['DM_num'] = df[f"{parameter}_DM"].map({'R':0,'A':1,'D':2})

        fig = px.scatter_geo(df, 
                            lat='latitude', lon='longitude',
                            color='DM_num',
                            color_continuous_scale='Viridis',
                            title=f"Global Data Mode for BGC Parameter: {parameter}")
        
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=f"bgc_map_{parameter}_{timestamp}.png"
        image_path = os.path.join("out_img", filename)
        fig.write_image(image_path)

        return f"✅ SUCCESS: Map generated and saved to {image_path}. Found {len(df)} profiles with {parameter} data."
    except ValueError as ve:
        return f"Input parsing error: {ve}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error generating map: {e}\nTraceback:\n{tb}"

inp= {'lat_min':5, 'lat_max': 25, 'lon_min': 80,
                    'lon_max': 100, 'plot': True, 'x': 'PRES', 'y': 'TEMP', 'plot_opt': 'scatter'}
# r=fetch_argo_data_by_region_plot(inp)
# print(r)



import numpy as np

def generate_data_analysis_summary(user_query: str) -> str:
    """
    Summarise the filtered ARGO dataset and ask the LLM for an
    oceanographic interpretation.  Keeps the prompt compact even
    when the DataFrame is large.
    """
    if not os.path.exists("argo_data.csv"):
        return ""
        
    df = pd.read_csv("argo_data.csv")
    if df.empty:
        return ""
    
    try:
        numeric_desc = df.describe(include=[np.number]).transpose()  # index = column names

        # Optional: include extra percentiles (10%, 90%) in describe
        extra_percentiles = df.describe(percentiles=[0.1, 0.9], include=[np.number]).transpose()

        # Convert describe() to a compact markdown table for the LLM
        stats_md = numeric_desc.round(4).to_markdown()

        # Add valid/null counts & mode as a small table
        modes = []
        for col in numeric_desc.index:
            try:
                mode_series = df[col].mode(dropna=True)
                mode_val = mode_series.iloc[0] if not mode_series.empty else ""
            except Exception:
                mode_val = ""
            valid_count = int(df[col].count())
            null_count = int(len(df) - valid_count)
            modes.append({"column": col, "mode": mode_val, "valid_count": valid_count, "null_count": null_count})

        modes_df = pd.DataFrame(modes).set_index("column")
        modes_md = modes_df.to_markdown()

        # Correlation matrix (useful for LLM to spot relationships)
        if len(numeric_desc.index) >= 2:
            corr_df = df[numeric_desc.index].corr().round(3)
            corr_md = corr_df.to_markdown()
        else:
            corr_md = ""

        # Compose the stats_text to include the describe table, modes, and correlations
        stats_text = "### DESCRIPTIVE STATISTICS (df.describe())\n\n"
        stats_text += stats_md + "\n\n"
        stats_text += "### MODE & COUNTS\n\n"
        stats_text += modes_md + "\n\n"
        if corr_md:
            stats_text += "### CORRELATION MATRIX\n\n"
            stats_text += corr_md + "\n\n"
        print(stats_text)
        # ---------- 2. Monthly aggregates (if DATE present) ----------
        monthly_text = ""
        date_col = "DATE"
        if date_col:
            # coerce to datetime (safe); you can pass dayfirst=True if your dates are D/M/Y
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

            # Drop rows without a valid datetime and set index
            monthly_df = df.dropna(subset=[date_col]).set_index(date_col)

            if not monthly_df.empty:
                # only aggregate columns that exist
                agg_cols = [c for c in ["TEMP", "PSAL", "PRES"] if c in monthly_df.columns]
                if agg_cols:
                    monthly_means = (
                        monthly_df
                        .resample("M")
                        .agg({c: "mean" for c in agg_cols})
                        .head(12)   # limit to first 12 months to keep size small
                    )
                    if not monthly_means.empty:
                        monthly_text = "### FIRST 12 MONTHLY MEANS (TEMP/PSAL/PRES)\n"
                        monthly_text += monthly_means.round(4).to_markdown() + "\n\n"

        print(monthly_text)
        print(monthly_text)
        # ---------- 3. Random sample of rows (max 200) ----------
        sample_text = ""
        if len(df) > 0:
            sample_df = df.sample(n=min(200, len(df)), random_state=42)
            # Keep only key columns to reduce size
            key_cols = [c for c in ["DATE", "LATITUDE", "LONGITUDE", "TEMP", "PSAL", "PRES"] if c in sample_df.columns]
            sample_text = "### RANDOM SAMPLE (up to 200 rows)\n"
            sample_text += sample_df[key_cols].to_markdown(index=False) + "\n\n"
        print(sample_text)
        # ---------- 4. Compose final prompt ----------
        analysis_prompt = f"""
        Based on the following user query and ARGO float data, provide a concise but
        scientifically sound oceanographic analysis.

        USER QUERY:
        {user_query}

        {stats_text}
        {monthly_text}
        {sample_text}

        Please discuss:
        1. What these statistics reveal about temperature, salinity, and pressure.
        2. Any seasonal trends or anomalies.
        3. How these values compare with typical ocean conditions for this region/time.
        """
        # Initialize LLM for analysis
        print(analysis_prompt)
        analysis_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        
        # Get analysis from LLM
        analysis_response = analysis_llm.invoke(analysis_prompt)
        
        return f"\n\n📊 **DATA ANALYSIS SUMMARY:**\n{analysis_response.content}"
    except Exception as e:
        return f"\n\n⚠️ Could not generate data analysis summary: {str(e)}"

# r=generate_data_analysis_summary("")
# print(r)
import requests
import base64
import pathlib
import os
def upload_image_to_imgbb(image_path, api_key):
    """Upload image to ImgBB and return public URL"""
    try:
        with open(image_path, "rb") as file:
            # Convert to base64
            image_data = base64.b64encode(file.read()).decode('utf-8')
        
        url = "https://api.imgbb.com/1/upload"
        payload = {
            'key': api_key,  # Get free API key from imgbb.com
            'image': image_data,
        }
        
        response = requests.post(url, data=payload)
        result = response.json()
        
        if result['success']:
            return result['data']['url']
        else:
            return None
    except Exception as e:
        print(f"Error uploading to ImgBB: {e}")
        return None

# IMGBB_API="a6866f39a2f6b16ae2d28d23d9f4a9aa"
# upload_image_to_imgbb(os.path.join("out_img", "plot_histogram_TEMP_None_20250926_102333.png"), IMGBB_API)


def generate_plot_summary_from_image(image_path: str):
    try:
        # Upload to ImgBB first
        imgbb_api_key = os.getenv("IMGBB_API")  # Add to your .env file
        public_url = upload_image_to_imgbb(image_path, imgbb_api_key)
        
        if not public_url:
            return "Could not upload image for analysis"
        
        vision_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        text_prompt = """
        You are an expert data analyst specializing in oceanography.
        Look at the attached plot from the Argo float program.
        Provide a brief, one-paragraph explanation of what this plot shows.
        - What are the variables on the axes?
        - What is the relationship, trend, or distribution shown?
        - What is the key insight a non-expert should take away from this visualization?
        Be precise and clear.
        """

        from langchain_core.messages import HumanMessage
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": public_url}},
            ]
        )
        response = vision_llm.invoke([message])
        explanation_text = response.content

        summary_path = pathlib.Path(image_path).with_suffix(".txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(explanation_text)

    except Exception as e:
        error_message = f"Could not generate summary for {os.path.basename(image_path)}: {e}"
        summary_path = pathlib.Path(image_path).with_suffix(".txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(error_message)
        print(error_message)
        return error_message

generate_plot_summary_from_image("out_img/plot_histogram_TEMP_None_20250926_102333.png")