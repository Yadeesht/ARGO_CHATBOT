# ARGO_CHATBOT: FloatChat 🌊 - ARGO Ocean Data Assistant

## Overview

**ARGO_CHATBOT** (FloatChat) is an interactive assistant that makes ocean float (ARGO) data accessible and interpretable for everyone—from ocean researchers to curious citizens and fishermen. It uses advanced language models and data visualization tools to answer questions, generate insights, and create ready-to-use maps and plots from global ARGO float datasets.

This project was built for rapid prototyping and public science, making complex ocean data easy to query and understand.

---

## Key Features

- **Conversational ARGO Data Analysis:** Ask questions in plain English about ocean temperature, salinity, depth, and more.
- **Automated Visualizations:** Instantly generates at least three high-quality, context-optimized plots (using Plotly, Matplotlib, Folium, Seaborn).
- **BGC Parameter Maps:** Visualize biogeochemical parameters like DOXY, CHLA, NITRATE, etc.
- **Smart Data Fetching:** Flexible region, time, and parameter filters with auto-throttling to avoid data overload.
- **Image Analysis:** Summarizes the meaning of generated plots using Google Gemini multimodal LLMs.
- **Streamlit Web Interface:** Modern chat-like UI with support for chat history, images, and deep research toggles.
- **Built with LangChain ReACT Agent:** Uses a tool-using agent to parse questions, call Python tools, and synthesize answers step-by-step.

---

## Tech Stack

- **Python 3**
- **Streamlit**: Frontend chat UI
- **LangChain ReACT Agent**: Reasoning and tool-use pipeline
- **Google Gemini (via LangChain)**: LLM for language and image analysis
- **argopy**: ARGO float data access
- **Plotly, Matplotlib, Seaborn, Folium**: Visualization

---

## Usage

### Quick Start

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Yadeesht/ARGO_CHATBOT.git
    cd ARGO_CHATBOT
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure API Keys:**
    - Create a `.env` file and add your Google Gemini and ImgBB API keys(ImgBB is to have the image at publically accessble to share it with LLM).
      ```
      GOOGLE_API_KEY=your_google_gemini_key
      IMGBB_API=your_imgbb_key
      ```

4. **Run the app:**
    ```bash
    streamlit run app.py
    ```
    - Open the Streamlit URL in your browser.

---

## How It Works

- Ask a question (e.g. "Show temperature profiles in the Bay of Bengal for 2022", "Plot dissolved oxygen in the Arabian Sea").
- The agent parses your query, fetches relevant ARGO data, and generates at least three smart visualizations.
- For each plot, you get a short, LLM-generated natural language summary.
- Toggle **Deep Research** for more advanced, slower queries.

---

## Example Queries

- "What is the temperature trend near Chennai over the past year?"
- "Show me profiles of salinity and pressure for floats in the North Atlantic."
- "Visualize nitrate (NITRATE) concentration for a specific region."
- "Explain the physical meaning of this plot."

---

## Data Schema

Available ARGO columns include:

- `CYCLE_NUMBER`, `DATA_MODE`, `DIRECTION`, `PLATFORM_NUMBER`, `POSITION_QC`
- `PRES`, `PSAL`, `TEMP`, `TIME`, `LATITUDE`, `LONGITUDE`
- Biogeochemical (BGC) parameters: `DOXY`, `CHLA`, `NITRATE`, `PH_IN_SITU_TOTAL`, etc.

---

## For Developers

- Main logic in `app.py`
- Data and plot tools are modularized (see `@tool` functions)
- Add new plot types or LLM tools by extending the agent's toolset
- See extensive comments in `app.py` for tool schemas, validation, and agent prompting

---

## Limitations
- The complete project is done within 48 hours so we didnt focus much on prompt tunning. work on that.
- visualization logic need to be tunned sometimes LLM gives wrong suggestion on plot that need to be generated and also after generating we are passing the image to the LLM for summary of the pic so u can remove the image by making LLM to flag the image. 
- Heavy queries may timeout; the agent will automatically reduce data scope and retry. 
- Not all float data may be available for all parameters or regions.
