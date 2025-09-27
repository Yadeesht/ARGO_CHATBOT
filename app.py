# app.py
import streamlit as st
import pandas as pd
import os
import json
import re
import traceback
from dotenv import load_dotenv
import sys
import shutil 
from datetime import datetime 
from argopy import ArgoIndex
from argopy.plot import scatter_map
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
import numpy as np
import pathlib
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
from langchain.prompts import PromptTemplate
import requests
import base64


from argopy import DataFetcher

# Load environment variables from .env file
load_dotenv()

# --- Helper: parse action input robustly ---
def parse_action_input(tool_input):
    """
    Parses the tool_input, which is expected to be a JSON string,
    potentially wrapped in markdown code blocks.
    """
    if isinstance(tool_input, dict):
        return tool_input

    s = str(tool_input).strip()
    
    # Clean the string if it's wrapped in markdown json block
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON from tool input: {s}. Error: {e}")

def parse_bgc_tool_input(tool_input):
    """
    Parses the tool_input for the BGC tool, which is expected to be a JSON string
    with a "parameter" key.
    """
    if isinstance(tool_input, dict):
        if "parameter" in tool_input:
            return tool_input["parameter"]
        else:
            raise ValueError("Input dictionary is missing the 'parameter' field.")

    s = str(tool_input).strip()

    # Clean the string if it's wrapped in markdown json block
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    try:
        obj = json.loads(s)
        return obj["parameter"]
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON from tool input: {s}. Error: {e}")

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
                        .resample("ME")
                        .agg({c: "mean" for c in agg_cols})
                        .head(12)   # limit to first 12 months to keep size small
                    )
                    if not monthly_means.empty:
                        monthly_text = "### FIRST 12 MONTHLY MEANS (TEMP/PSAL/PRES)\n"
                        monthly_text += monthly_means.round(4).to_markdown() + "\n\n"

        # ---------- 3. Random sample of rows (max 200) ----------
        sample_text = ""
        if len(df) > 0:
            sample_df = df.sample(n=min(200, len(df)), random_state=42)
            # Keep only key columns to reduce size
            key_cols = [c for c in ["DATE", "LATITUDE", "LONGITUDE", "TEMP", "PSAL", "PRES"] if c in sample_df.columns]
            sample_text = "### RANDOM SAMPLE (up to 200 rows)\n"
            sample_text += sample_df[key_cols].to_markdown(index=False) + "\n\n"

        # ---------- 4. Compose final prompt ----------
        analysis_prompt = f"""
        Based on the following user query and ARGO float data, provide a concise but
        scientifically sound oceanographic analysis.While also keeping the scientifically sound ,generate an another version of the analysis which is easy to understand for a non expert user.
        This should be appended to the scientifically sound and detailed explanation.

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
        analysis_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        
        # Get analysis from LLM
        analysis_response = analysis_llm.invoke(analysis_prompt)
        
        return f"\n\n📊 **DATA ANALYSIS SUMMARY:**\n{analysis_response.content}"
    except Exception as e:
        return f"\n\n⚠️ Could not generate data analysis summary: {str(e)}"

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
    
# --- Tools ---
@tool()
def fetch_argo_data_by_region_plot(tool_input):
    """
    Fetch ARGO data and optionally generate high-quality visualizations.

    INPUT (JSON)
    - either {"lat":..,"lon":..} OR bounding box {"lat_min":..,"lat_max":..,"lon_min":..,"lon_max":..}
    - platform_number, cycle_number, temp_min, temp_max, pres_min, pres_max, date_start, date_end, data_mode, etc. as a parameter.
    - if plot is requested or suitable then:
        "plot": true|false
        "plot_print": [   # REQUIRED when plot=true (see rules)
            { "type": "<plotly_type>", "x": "<column>", "y": "<column>", ...optional params... },
            ...
        ]
    - {"DR": True} if Deep research is asked by the user. else False.

    AVAILABLE COLUMNS (argo_params)
    ["CYCLE_NUMBER","DATA_MODE","DIRECTION","PLATFORM_NUMBER","POSITION_QC",
    "PRES","PRES_ERROR","PRES_QC","PSAL","PSAL_ERROR","PSAL_QC",
    "TEMP","TEMP_ERROR","TEMP_QC","TIME_QC","LATITUDE","LONGITUDE","TIME"]

    GENERAL RULES (strict)
    1. If "plot": true → "plot_print" must be present and must contain **at least 3 valid plot objects** unless the fetched data cannot be meaningfully visualized.
    2. Each plot object must be a dict with "type" (one of supported plotly types) and the columns ("x" and/or "y") required by that plot type.
    3. Validate columns exist in the fetched dataframe before plotting. If a requested column is missing, skip that plot and explain why.
    4. Do not propose redundant plots: if two plot types convey essentially the same view (e.g., "strip" vs "scatter" for the same numeric pair and grouping), include only one.
    5. When plot generation times out or a network error occurs, first reduce date range; if still failing, reduce bounding box. Return the attempted change and retry.
    6. If there is a timeout error while fetching the data,then first prioritize decreasing the date range to fetch the data to a smaller range and then if the error still persists then prioritize decreasing the bounding box to a smaller box. if it encounters more than twice then aggressively reduce the parameters.


    COLUMN USAGE GUIDELINES
    - Numeric / continuous (suitable for x/y): PRES, PRES_ERROR, PSAL, PSAL_ERROR, TEMP, TEMP_ERROR, LATITUDE*, LONGITUDE*, CYCLE_NUMBER*, PLATFORM_NUMBER*
    * LATITUDE/LONGITUDE are numeric but should only be used together for geoplots (see below)
    * CYCLE_NUMBER / PLATFORM_NUMBER are primarily for grouping/coloring, not continuous trends
    - Date: TIME → allowed as x (for time series), not as y
    - Categorical: DATA_MODE, DIRECTION, POSITION_QC, *_QC → use for color/facet/x (bar/box/violin/histogram), not for continuous y in line plots

    PLOT TYPE → REQUIRED COLUMNS / RECOMMENDED USAGE (examples)
    - ******Dont take a random guess on chossing the x and y becuase the plot will not make any sense so be carefull******
    - never do a line plot using TEMP vs PSAL or TEMP vs PRES
    - scatter, line: numeric x & y (e.g., TEMP vs PSAL; TEMP vs PRES). For line representing vertical profiles: **group by CYCLE_NUMBER or PLATFORM_NUMBER and sort by PRES**; reverse y-axis so depth increases downward.
    - histogram, density_heatmap: single numeric x (histogram), numeric x & y pairs for density_heatmap.
    - box, violin: categorical x (DATA_MODE / PLATFORM_NUMBER / CYCLE_NUMBER) and numeric y (TEMP/PSAL).
    - scatter_geo, scatter_mapbox, choropleth*, density_mapbox: require LATITUDE & LONGITUDE (and value column for choropleth).
    - parallel_coordinates, parallel_categories: multivariate views of several numeric/categorical vars.
    - imshow: 2D matrix (only if user provides gridded data or derived matrix, otherwise skip).

    CHOOSING THE BEST PLOTS (heuristics; agent must follow)
    1. Pick plots that match data types (continuous vs categorical vs geospatial).
    2. Always include at least one distribution plot (histogram/box/violin) for a main numeric variable (TEMP or PSAL).
    3. If spatial coverage exists, include one geospatial view (scatter_geo or density_mapbox) when LAT/LON present.
    4. If profiles exist (CYCLE_NUMBER or PLATFORM_NUMBER present and PRES available), include one profile view: TEMP vs PRES grouped by CYCLE_NUMBER (line per profile) OR scatter of TEMP vs PRES.
    5. Add one relational plot (e.g., TEMP vs PSAL scatter / density) to show physical relationships (T–S diagram).
    6. Minimum = 3 plots; if user supplied more valid specs, generate all valid and non-redundant ones.

    VALIDATION BEFORE PLOTTING
    - Confirm each plot's required columns exist and are numeric where needed.
    - For line/profile plots: ensure grouping exists (CYCLE_NUMBER or PLATFORM_NUMBER) and sort by PRES per group; otherwise convert to scatter.
    - For geo plots: require both LATITUDE and LONGITUDE.

    RETURN
    - Return a list of generated image paths and brief notes describing which plots were created and any skipped due to invalid columns or redundancy.

    EXAMPLE plot_print:
    "plot_print": [
    {"type":"scatter","x":"TEMP","y":"PSAL"},
    {"type":"scatter_geo","x":"LONGITUDE","y":"LATITUDE"},
    {"type":"line","x":"TEMP","y":"PRES"}   # agent must group by CYCLE_NUMBER when making this line plot
    ]
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

        query_params["date_start"] = parsed.get("date_start", "2023-01-01")
        query_params["date_end"] = parsed.get("date_end", "2024-12-31")

        box_numeric = [
            query_params["lon_min"], query_params["lon_max"],
            query_params["lat_min"], query_params["lat_max"],
            query_params["pres_min"], query_params["pres_max"],
            query_params["date_start"], query_params["date_end"]
        ]
        try:
            if "DR" in parsed and parsed["DR"]==True:
                ds = DataFetcher(fs_opts={'timeout': 150}).region(box_numeric).to_xarray()
            else:
                ds = DataFetcher(fs_opts={'timeout': 60}).region(box_numeric).to_xarray()  
            df = ds.to_dataframe()
        except Exception as e:
            return f"Error fetching data: {e} most likely the amount of data we are fetching is too much so try to reduce the param which we are controlling"

        if df.empty:
            return "No data found for the specified region and time."
        filters_applied=[]
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

        param_stats = []
        if "TIME" in df_reset.columns:
            df_reset['DATE'] = pd.to_datetime(df_reset['TIME']).dt.date
            df_reset = df_reset.groupby('DATE').first().reset_index()
            min_date = pd.to_datetime(df_reset["DATE"].min())
            max_date = pd.to_datetime(df_reset["DATE"].max())
            param_stats.append(f"Date between {min_date} and {max_date}")
        

        # Parameter ranges
        if "TEMP" in df_reset.columns:
            temp_range = f"{df_reset['TEMP'].min():.2f} to {df_reset['TEMP'].max():.2f}°C"
            param_stats.append(f"🌡️ Temperature: {temp_range}")
        
        if "PSAL" in df_reset.columns:
            psal_range = f"{df_reset['PSAL'].min():.2f} to {df_reset['PSAL'].max():.2f}"
            param_stats.append(f"🧂 Salinity: {psal_range}")
            
        if "PRES" in df_reset.columns:
            pres_range = f"{df_reset['PRES'].min():.1f} to {df_reset['PRES'].max():.1f} dbar"
            param_stats.append(f"📊 Pressure: {pres_range}")
        
        df_reset.to_csv("argo_data.csv",index=False)
        

        if parsed["plot"]==True:
            plot_item = parsed.get("plot_print")
            plot_message=""  
            fig=None
            for spec in plot_item:
                plot = spec.get("type")
                x  = spec.get("x")
                y  = spec.get("y")
                if plot == "scatter":
                    fig = px.scatter(data_frame=df_reset, x=x, y=y, title=f"ARGO Data: {x} vs. {y}")
                elif plot == "scatter_3d":
                    z = parsed.get("z", "PRES")
                    fig = px.scatter_3d(data_frame=df_reset, x=x, y=y, z=z, title=f"ARGO 3D: {x} vs. {y} vs. {z}")
                elif plot == "scatter_polar":
                    fig = px.scatter_polar(data_frame=df_reset, r=x, theta=y, title=f"ARGO Polar: {x} vs. {y}")
                elif plot == "scatter_ternary":
                    a = parsed.get("a", x)
                    b = parsed.get("b", y)
                    c = parsed.get("c", "PRES")
                    fig = px.scatter_ternary(data_frame=df_reset, a=a, b=b, c=c, title=f"ARGO Ternary: {a}-{b}-{c}")
                elif plot == "line":
                    fig = px.line(data_frame=df_reset, x=x, y=y, title=f"ARGO Data: {x} vs. {y}")
                elif plot == "line_3d":
                    z = parsed.get("z", "PRES")
                    fig = px.line_3d(data_frame=df_reset, x=x, y=y, z=z, title=f"ARGO 3D Line: {x} vs. {y} vs. {z}")
                elif plot == "line_polar":
                    fig = px.line_polar(data_frame=df_reset, r=x, theta=y, title=f"ARGO Polar Line: {x} vs. {y}")
                elif plot == "area":
                    fig = px.area(data_frame=df_reset, x=x, y=y, title=f"ARGO Area: {x} vs. {y}")
                elif plot == "bar":
                    fig = px.bar(data_frame=df_reset, x=x, y=y, title=f"ARGO Bar: {x} vs. {y}")
                elif plot == "histogram":
                    fig = px.histogram(data_frame=df_reset, x=x, title=f"ARGO Histogram: {x}")
                elif plot == "violin":
                    fig = px.violin(data_frame=df_reset, y=y, title=f"ARGO Violin: {y}")
                elif plot == "box":
                    fig = px.box(data_frame=df_reset, y=y, title=f"ARGO Box Plot: {y}")
                elif plot == "strip":
                    fig = px.strip(data_frame=df_reset, y=y, title=f"ARGO Strip Plot: {y}")
                elif plot == "pie":
                    fig = px.pie(data_frame=df_reset, names=x, values=y, title=f"ARGO Pie: {x}")
                elif plot == "sunburst":
                    path = parsed.get("path", [x])
                    fig = px.sunburst(data_frame=df_reset, path=path, values=y, title=f"ARGO Sunburst")
                elif plot == "treemap":
                    path = parsed.get("path", [x])
                    fig = px.treemap(data_frame=df_reset, path=path, values=y, title=f"ARGO Treemap")
                elif plot == "icicle":
                    path = parsed.get("path", [x])
                    fig = px.icicle(data_frame=df_reset, path=path, values=y, title=f"ARGO Icicle")
                elif plot == "funnel":
                    fig = px.funnel(data_frame=df_reset, x=x, y=y, title=f"ARGO Funnel: {x} vs. {y}")
                elif plot == "funnel_area":
                    fig = px.funnel_area(data_frame=df_reset, names=x, values=y, title=f"ARGO Funnel Area")
                elif plot == "density_contour":
                    fig = px.density_contour(data_frame=df_reset, x=x, y=y, title=f"ARGO Density Contour: {x} vs. {y}")
                elif plot == "density_heatmap":
                    fig = px.density_heatmap(data_frame=df_reset, x=x, y=y, title=f"ARGO Density Heatmap: {x} vs. {y}")
                elif plot == "scatter_geo":
                    if "LATITUDE" in df_reset.columns and "LONGITUDE" in df_reset.columns:
                        fig = px.scatter_geo(data_frame=df_reset, lat="LATITUDE", lon="LONGITUDE", color=parsed.get("color", y), title=f"ARGO Geographic Scatter")
                    else:
                        plot_message += f"❌ Geographic coordinates (LATITUDE, LONGITUDE) not available for scatter_geo. "
                        continue
                elif plot == "choropleth":
                    locations = parsed.get("locations", "PLATFORM_NUMBER")
                    fig = px.choropleth(data_frame=df_reset, locations=locations, color=y, title=f"ARGO Choropleth")
                elif plot == "choropleth_mapbox":
                    if "LATITUDE" in df_reset.columns and "LONGITUDE" in df_reset.columns:
                        fig = px.choropleth_mapbox(data_frame=df_reset, lat="LATITUDE", lon="LONGITUDE", color=y, title=f"ARGO Choropleth Mapbox")
                    else:
                        plot_message += f"❌ Geographic coordinates not available for choropleth_mapbox. "
                        continue
                elif plot == "scatter_mapbox":
                    if "LATITUDE" in df_reset.columns and "LONGITUDE" in df_reset.columns:
                        fig = px.scatter_mapbox(data_frame=df_reset, lat="LATITUDE", lon="LONGITUDE", color=parsed.get("color", y), title=f"ARGO Scatter Mapbox")
                    else:
                        plot_message += f"❌ Geographic coordinates not available for scatter_mapbox. "
                        continue
                elif plot == "density_mapbox":
                    if "LATITUDE" in df_reset.columns and "LONGITUDE" in df_reset.columns:
                        fig = px.density_mapbox(data_frame=df_reset, lat="LATITUDE", lon="LONGITUDE", title=f"ARGO Density Mapbox")
                    else:
                        plot_message += f"❌ Geographic coordinates not available for density_mapbox. "
                        continue
                elif plot == "parallel_coordinates":
                    dimensions = parsed.get("dimensions", ["TEMP", "PSAL", "PRES"])
                    fig = px.parallel_coordinates(data_frame=df_reset, dimensions=dimensions, title=f"ARGO Parallel Coordinates")
                elif plot == "parallel_categories":
                    dimensions = parsed.get("dimensions", [x,y])
                    fig = px.parallel_categories(data_frame=df_reset, dimensions=dimensions, title=f"ARGO Parallel Categories")
                elif plot == "imshow":
                    try:
                        pivot_data = df_reset.pivot_table(values=y, index=x, columns=parsed.get("columns", "TIME"))
                        fig = px.imshow(pivot_data, title=f"ARGO Heatmap: {y}")
                    except Exception as e:
                        plot_message += f"❌ Cannot create imshow plot: {e}. Try using density_heatmap instead. "
                        continue
                else:
                    plot_message += f"❌ Unable to find the {parsed['plot_print']} option. Available options: scatter, scatter_3d, scatter_polar, scatter_ternary, line, line_3d, line_polar, area, bar, histogram, violin, box, strip, pie, sunburst, treemap, icicle, funnel, funnel_area, density_contour, density_heatmap, scatter_geo, choropleth, choropleth_mapbox, scatter_mapbox, density_mapbox, parallel_coordinates, parallel_categories, imshow. "            

                if fig:     
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"plot_{plot}_{x}_{y}_{timestamp}.png"
                    image_path = os.path.join("out_img",filename)
                    fig.write_image(image_path)
                    generate_plot_summary_from_image(image_path)
                    plot_message += f"Plot {plot} saved to {image_path}."

        # --- Folium Interactive Map Generation ---
        map_message = ""
        if "LATITUDE" in df_reset.columns and "LONGITUDE" in df_reset.columns and not df_reset.empty:
            try:
                # Create a folium map centered on the mean coordinates of the data
                map_center_lat = df_reset["LATITUDE"].mean()
                map_center_lon = df_reset["LONGITUDE"].mean()
                
                m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=4)

                # Add a marker for the center
                folium.Marker(
                    [map_center_lat, map_center_lon], 
                    popup=f"Approx. center of {len(df_reset)} data points"
                ).add_to(m)

                # Save map to an HTML file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                map_filename = f"interactive_map_{timestamp}.html"
                map_path = os.path.join("out_img", map_filename)
                m.save(map_path)
                map_message = f" Interactive map saved to {map_path}."
            except Exception as e:
                map_message = f" Could not generate interactive map: {e}"
        # --- End Folium ---

        return f"✅ Plots generated in folder out_img.{plot_message}{map_message}"
    
    except ValueError as ve:
        return f"Input parsing error: {ve}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error fetching data: {e}\nTraceback:\n{tb}"

@tool()
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
        generate_plot_summary_from_image(image_path)

        return f"✅ SUCCESS: Map generated and saved to {image_path}. Found {len(df)} profiles with {parameter} data."
    except ValueError as ve:
        return f"Input parsing error: {ve}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error generating map: {e}\nTraceback:\n{tb}"

# --- LLM + Agent setup ---

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Define the list of tools available to the agent
tools = [fetch_argo_data_by_region_plot, generate_bgc_parameter_map]

# Improve the prompt to ask the agent to emit JSON for action inputs
react_prompt_template = '''Answer the following questions as best you can. Your final answer should be a comprehensive summary of your findings, incorporating details from observations you made. You have access to the following tools:

{tools}

When you call a tool, the "Action Input" MUST be a valid JSON object matching the tool's arguments.

TOOL USAGE SUMMARY (quick)

1) fetch_argo_data_by_region
   - Accepts geographic, parameter, time, depth, platform, cycle and quality filters.
   - Geographic: single point {{ "lat": <num>, "lon": <num>, "buffer": <deg> }} OR bounding box {{ "lat_min": <num>, "lat_max": <num>, "lon_min": <num>, "lon_max": <num> }}.
   - Parameter filters: "pres_min", "pres_max", "temp_min", "temp_max", "psal_min", "psal_max".
   - Time: "date_start","date_end" formatted "YYYY-MM-DD".
   - Platform / cycle: "platform_number", "cycle_number".
   - Data mode: "data_mode" ("R","A","D").
   - For this tools have a {{"DR":True}} if Deep research is asked by the user.
   - If the query is too large and fetch times out, PRIORITY for retry: first reduce date range, then reduce bounding box. Report any automatic reductions you attempt.

2) generate_bgc_parameter_map
   - Requires: {{ "parameter": "<BGC_PARAM>" }}.
   - Available BGC parameters: "DOXY","BBP700","BBP470","CHLA","NITRATE","PH_IN_SITU_TOTAL","DOWNWELLING_PAR".

DATA SCHEMA (available columns)
["CYCLE_NUMBER","DATA_MODE","DIRECTION","PLATFORM_NUMBER","POSITION_QC",
 "PRES","PRES_ERROR","PRES_QC","PSAL","PSAL_ERROR","PSAL_QC",
 "TEMP","TEMP_ERROR","TEMP_QC","TIME_QC","LATITUDE","LONGITUDE","TIME"]

Type hints:
- Numeric / continuous: PRES, PRES_ERROR, PSAL, PSAL_ERROR, TEMP, TEMP_ERROR, LATITUDE, LONGITUDE (and TIME if converted to timestamp)
- Categorical / discrete: DATA_MODE, DIRECTION, POSITION_QC, *_QC, PLATFORM_NUMBER, CYCLE_NUMBER (can be used for grouping/colouring but not always continuous)
- TIME: allowed as x for time-series only (convert to datetime), not as y.

Requirements for plotting:
1. If "plot": true → "plot_print" is REQUIRED and must include at least **3 valid plot objects** unless the data cannot be meaningfully visualized (explain why).
2. Each plot object must specify "type" (a Plotly Express type) and the columns it requires ("x" and/or "y") appropriate for that plot type.
3. Validate columns exist in the fetched dataframe before plotting. If a requested column is missing, skip that plot and add an explanation to the observation.
4. Do **not** propose redundant plots. If two plot types convey essentially the same insight for the same columns (e.g., "strip" vs "scatter" with the same encoding), include only one.
5. For geo plots ("scatter_geo","scatter_mapbox","choropleth_mapbox","density_mapbox"), **both** LATITUDE and LONGITUDE must be present and appropriate.
6. For line/profile plots where PRES is used as the vertical axis:
   - Treat this as a vertical profile only if you can group by CYCLE_NUMBER or PLATFORM_NUMBER.
   - Sort each group by PRES and reverse the y-axis (depth increases downward).
   - If grouping or PRES sort is impossible, convert to a scatter plot instead of a global line.
7. TIME may be used as x for time-series plots; convert to datetime and ensure the times are contiguous/ordered for lines.
8. Categorical fields (DATA_MODE, DIRECTION, *_QC) should be used for color, facet, or x in box/violin/bar plots — not as y in continuous line charts.

Plot types you may pick from (examples): "scatter","line","histogram","box","violin","density_heatmap","scatter_geo","scatter_mapbox","parallel_coordinates","imshow", etc. Choose plot types that match the data types available.

HOW TO CHOOSE THE BEST 3 (Agent heuristics — follow these in order)

When the user requests plots (or when the agent suggests them), choose at least three plots that maximize insight and minimize redundancy:

1. Mandatory: include a **distribution view** for a primary numeric variable (TEMP or PSAL) — e.g., histogram or box/violin.
2. If LAT/LON present: include **one geospatial view** (scatter_geo or density_mapbox) showing coverage.
3. If PRES + CYCLE_NUMBER (or PLATFORM_NUMBER) present: include **one profile view** (TEMP vs PRES grouped by cycle or platform). Use line per group (sorted by PRES) — otherwise use scatter.
4. Include a **relational view** (e.g., TEMP vs PSAL scatter or density) to show physical relationships (T–S diagram).
5. If more than 3 meaningful, you may include additional non-redundant plots the user requested.
6. If a user provided explicit "plot_print", validate & honor it; if any plots are invalid or missing required fields, suggest corrections and generate the valid subset (still aiming for 3 if feasible).


VALIDATION BEFORE CALLING PLOT TOOL (what you must check)

- Column existence in df.
- Column types: numeric required for continuous axes.
- For geo plots: require LATITUDE & LONGITUDE.
- For line/profile: group & sort by PRES present; else fallback to scatter.
- Convert TIME to datetime when used as x.
- If plot objects < 3 after validation, explain why and either:
  - generate the valid ones and note why you couldn't reach 3, or
  - request clarification from the user.

IMPORTANT STOPPING CONDITIONS

- If you receive a "✅ SUCCESS" observation from any tool, immediately proceed to "Final Answer".
- If a map/visualization is generated successfully, include the image path in "Final Answer" and DO NOT repeat the same action.
- Do NOT call the same tool multiple times with the same parameters in a loop.
- Use the most appropriate parameters based on the user's question and these heuristics.

CALL / RESPONSE FORMAT (strict)

Follow this exact format for your reasoning and calls:

Question: {input}
Thought: (brief reasoning about what you will do next)
Action: one of [{tool_names}]
Action Input: (valid JSON matching the tool's schema)
Observation: (tool output)
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: (comprehensive, include observations, list of generated image paths, and any adjustments made)

Begin!

Question: {input}
Thought:{agent_scratchpad}'''



prompt = PromptTemplate.from_template(react_prompt_template)

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the Agent Executor which will run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,max_iterations=10,early_stopping_method="force",handle_parsing_errors=True,return_intermediate_steps=True)


# --- Streamlit frontend ---nb 

st.title("FloatChat 🌊 - ARGO Ocean Data Assistant")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display images if present
        if "images" in message:
            for img_path in message["images"]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Generated: {os.path.basename(img_path)}", use_container_width=True)
        
        # Display HTML files if present
        if "html_files" in message:
            for html_path in message["html_files"]:
                if os.path.exists(html_path):
                    try:
                        with open(html_path, 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=400)  # Smaller in history
                    except Exception as e:
                        st.warning(f"Could not redisplay map: {os.path.basename(html_path)}")
        
        # Display text files if present
        if "txt_files" in message:
            for txt_path in message["txt_files"]:
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r') as f:
                            txt_content = f.read()
                        with st.expander(f"📋 {os.path.basename(txt_path)}", expanded=False):
                            st.text(txt_content)
                    except Exception as e:
                        st.warning(f"Could not redisplay text: {os.path.basename(txt_path)}")

# Create input area with checkbox
input_container = st.container()

with input_container:
    col1, col2 = st.columns([4, 1])
    
    with col1:
        prompt_text = st.chat_input("Ask about ARGO float data...")
    
    with col2:
        deep_research = st.checkbox("🔬 Deep Research", value=False)    

# Handle user input
if prompt_text :

    print(f"Deep Research Mode: {deep_research}")  # For debugging
    if os.path.exists("out_img"):
        shutil.rmtree("out_img")
    os.makedirs("out_img", exist_ok=True)

    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Get assistant response
    prompt_text += " Deep research=True." if deep_research else " Deep research=False."
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": prompt_text})
            response_text = response.get("output", "I encountered an error.")
            
            # Add analysis summary
            analysis_summary = generate_data_analysis_summary(prompt_text)
            if analysis_summary:
                response_text += analysis_summary

            st.markdown(response_text)
            
            # 🚀 Collect ALL files at once
            valid_images = []
            html_files = []
            txt_files = []
            
            if os.path.exists("out_img"):
                for filename in os.listdir("out_img"):
                    file_path = os.path.join("out_img", filename)
                    
                    if filename.endswith(('.png', '.jpg', '.jpeg')):
                        valid_images.append(file_path)
                    elif filename.endswith('.html'):
                        html_files.append(file_path)
                    elif filename.endswith('.txt'):
                        txt_files.append(file_path)

            # Display all content types
            displayed_content = []
            
            # Display images with their corresponding text analysis
            if valid_images:
                st.write(f"🖼️ **Generated {len(valid_images)} visualization(s):**")
                
                if len(valid_images) == 1:
                    st.image(valid_images[0], caption=f"Generated: {os.path.basename(valid_images[0])}", use_container_width=True)
                    # Display corresponding text file content
                    txt_path = pathlib.Path(valid_images[0]).with_suffix(".txt")
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r') as f:
                                txt_content = f.read()
                            st.write(f"**📋 Analysis:**")
                            st.text(txt_content)
                        except Exception as e:
                            st.warning(f"Could not display analysis for {os.path.basename(valid_images[0])}")
                elif len(valid_images) <= 3:
                    cols = st.columns(len(valid_images))
                    for i, img_path in enumerate(valid_images):
                        with cols[i]:
                            st.image(img_path, caption=f"{os.path.basename(img_path)}", use_container_width=True)
                            # Display corresponding text file content
                            txt_path = pathlib.Path(img_path).with_suffix(".txt")
                            if os.path.exists(txt_path):
                                try:
                                    with open(txt_path, 'r') as f:
                                        txt_content = f.read()
                                    st.write(f"**📋 Analysis:**")
                                    st.text(txt_content)
                                except Exception as e:
                                    st.warning(f"Could not display analysis")
                else:
                    for i in range(0, len(valid_images), 3):
                        cols = st.columns(3)
                        for j, img_path in enumerate(valid_images[i:i+3]):
                            with cols[j]:
                                st.image(img_path, caption=f"{os.path.basename(img_path)}", use_container_width=True)
                                # Display corresponding text file content
                                txt_path = pathlib.Path(img_path).with_suffix(".txt")
                                if os.path.exists(txt_path):
                                    try:
                                        with open(txt_path, 'r') as f:
                                            txt_content = f.read()
                                        st.write(f"**📋 Analysis:**")
                                        st.text(txt_content)
                                    except Exception as e:
                                        st.warning(f"Could not display analysis")
                
                displayed_content.append(f"Images: {', '.join([os.path.basename(f) for f in valid_images])}")
            
            # Display HTML files info (actual display happens in chat history)
            if html_files:
                st.write(f"🗺️ **Generated {len(html_files)} Interactive Map(s):**")
                for html_path in html_files:
                    st.success(f"📍 Map: {os.path.basename(html_path)} (displayed below)")
                
                displayed_content.append(f"Maps: {', '.join([os.path.basename(f) for f in html_files])}")
            
            final_content = response_text

            final_content += f"\n\n📎 **Visualizations : **"
            message_entry = {
                "role": "assistant", 
                "content": final_content
            }
            
            # Add file references for display in chat history
            if valid_images:
                message_entry["images"] = valid_images
            if html_files:
                message_entry["html_files"] = html_files
            if txt_files:
                message_entry["txt_files"] = txt_files
                
            st.session_state.messages.append(message_entry)

            if os.path.exists("argo_data.csv"):
                with open("argo_data.csv", "rb") as f:
                    st.download_button(
                        label="Download Last Queried Data (CSV)",
                        data=f,
                        file_name="argo_data.csv",
                        mime="text/csv"
                    )