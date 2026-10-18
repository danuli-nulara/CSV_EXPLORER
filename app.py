"""
CSV Explorer - Gradio edition (runs on Hugging Face Spaces)
-----------------------------------------------------------
Same analysis as the Tkinter version:
paste a dataset link or upload a file, then see the overview, columns,
statistics, categories, preview and charts.

Optional: AI insights from Google Gemini. To switch them on, add a Space
secret called GEMINI_API_KEY (Settings > Variables and secrets).
"""

# ---------- imports ----------
import io                      # treat downloaded bytes like a file
import os                      # file path helpers, environment variables
import re                      # recognise Google Drive / Sheets links
import tempfile                # a safe place to save the report file
import urllib.request          # download a dataset from a link

import gradio as gr
import matplotlib
matplotlib.use("Agg")          # draw charts without a window (a server has no screen)
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# ---------- ZeroGPU workaround ----------
# A free Hugging Face account can only run a Gradio Space on "ZeroGPU" hardware,
# and ZeroGPU refuses to start unless the app contains one @spaces.GPU function.
# This app doesn't need a GPU, so the function below does nothing and is never
# called (no GPU is ever used). It only keeps the Space from being stopped.
# The try/except lets the same file still run on your own computer, where the
# "spaces" package doesn't exist.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return None
except ImportError:
    pass

# ---------- settings ----------
MAX_MB = 50                                                  # biggest file we accept
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # change in Space variables if needed

# ---------- colour palette (same as the Tkinter version) ----------
BLACK = "#140C0E"          # main background
PANEL = "#2A1816"          # dark coffee (cards, tables)
COFFEE = "#6F4E37"         # medium coffee (borders)
BURGUNDY = "#6D1A2D"       # deep burgundy (header, buttons)
BURGUNDY_LIGHT = "#9B2B4A"  # brighter burgundy (chart bars)
BUTTER = "#FFE8A3"         # butter yellow (main text)
BUTTER_DIM = "#CDB988"     # softer butter (secondary text)

MISSING_CMAP = ListedColormap([COFFEE, BUTTER])                      # present / missing
CORR_CMAP = LinearSegmentedColormap.from_list(
    "burgundy_butter", [BURGUNDY_LIGHT, PANEL, BUTTER])              # -1 ... 0 ... +1

CHART_TYPES = [
    "Missing values heatmap",
    "Missing values per column",
    "Histogram + boxplot",
    "Top 10 categories",
    "Correlation heatmap",
    "Scatter plot",
]

CSS = f"""
:root, .dark, .light {{
    --body-background-fill: {BLACK};
    --background-fill-primary: {PANEL};
    --background-fill-secondary: {BLACK};
    --block-background-fill: {PANEL};
    --panel-background-fill: {PANEL};
    --block-border-color: {COFFEE};
    --border-color-primary: {COFFEE};
    --border-color-accent: {BUTTER};
    --body-text-color: {BUTTER};
    --body-text-color-subdued: {BUTTER_DIM};
    --block-label-text-color: {BUTTER_DIM};
    --block-label-background-fill: {PANEL};
    --block-title-text-color: {BUTTER_DIM};
    --input-background-fill: {BLACK};
    --input-border-color: {COFFEE};
    --input-placeholder-color: {COFFEE};
    --button-primary-background-fill: {BURGUNDY};
    --button-primary-background-fill-hover: {BURGUNDY_LIGHT};
    --button-primary-text-color: {BUTTER};
    --button-secondary-background-fill: {PANEL};
    --button-secondary-background-fill-hover: {COFFEE};
    --button-secondary-text-color: {BUTTER};
    --table-even-background-fill: {PANEL};
    --table-odd-background-fill: #342019;
    --table-border-color: {COFFEE};
    --color-accent: {BURGUNDY_LIGHT};
    --color-accent-soft: {PANEL};
    --link-text-color: {BUTTER};
}}
.gradio-container {{ background: {BLACK} !important; }}
.app-title {{ background: {BURGUNDY}; padding: 18px 24px; border-radius: 8px; }}
.app-title h1 {{ font-family: Georgia, serif; font-size: 2rem; margin: 0; color: {BUTTER}; }}
.app-title p {{ font-family: Georgia, serif; font-style: italic; margin: 4px 0 0; color: {BUTTER_DIM}; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
.card {{ background: {PANEL}; border: 1px solid {COFFEE}; border-radius: 6px;
         padding: 16px; text-align: center; }}
.card .value {{ font-family: Georgia, serif; font-size: 1.7rem; font-weight: bold; color: {BUTTER}; }}
.card .label {{ color: {BUTTER_DIM}; font-size: 0.9rem; }}
button[role="tab"].selected, button[role="tab"][aria-selected="true"] {{
    color: {BUTTER} !important; border-color: {BURGUNDY_LIGHT} !important; }}
@media (max-width: 700px) {{ .cards {{ grid-template-columns: repeat(2, 1fr); }} }}
"""


# =====================================================================
#  Helper functions (data work only, no screen stuff)
# =====================================================================
def fix_link(url):
    """Turn 'share' links into direct download links."""
    url = url.strip()

    # GitHub page link -> raw file link
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com")
        url = url.replace("/blob/", "/")

    # Google Drive share link -> direct download link
    match = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if match:
        url = f"https://drive.google.com/uc?export=download&id={match.group(1)}"

    # Google Sheets link -> CSV export link
    match = re.search(r"docs\.google\.com/spreadsheets/d/([^/]+)", url)
    if match:
        url = f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv"

    # Dropbox share link -> direct download
    if "dropbox.com" in url:
        url = url.replace("dl=0", "dl=1")

    return url


def read_dataset(source):
    """Load a CSV from a web link or from a file path and return a DataFrame."""
    source = source.strip().strip('"')
    limit = MAX_MB * 1024 * 1024

    # ---- case 1: it's a web link ----
    if source.lower().startswith(("http://", "https://")):
        url = fix_link(source)
        # some sites block Python, so we pretend to be a normal browser
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(limit + 1)       # read a little more than the limit

        if len(raw) > limit:
            raise ValueError(f"That file is bigger than {MAX_MB} MB.")

        # if we got a web page instead of a CSV, tell the user clearly
        start = raw[:500].lower()
        if b"<!doctype html" in start or b"<html" in start:
            raise ValueError(
                "That link gave back a web page, not a CSV file.\n"
                "Please use a direct link to the raw .csv file."
            )
        df = pd.read_csv(io.BytesIO(raw), encoding_errors="replace")

    # ---- case 2: it's an uploaded file ----
    else:
        if not os.path.exists(source):
            raise FileNotFoundError("Please paste a link starting with http:// or https://, or upload a file.")
        if os.path.getsize(source) > limit:
            raise ValueError(f"That file is bigger than {MAX_MB} MB.")
        df = pd.read_csv(source, encoding_errors="replace")

    if df.shape[1] == 0:
        raise ValueError("The file seems to be empty.")
    df.columns = df.columns.astype(str)          # make sure every column name is text
    return df


def split_columns(df):
    """Return (numerical column names, categorical column names)."""
    numeric = []
    categorical = []
    for col in df.columns:
        is_number = pd.api.types.is_numeric_dtype(df[col])
        is_bool = pd.api.types.is_bool_dtype(df[col])
        if is_number and not is_bool:
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def short(text, length=18):
    """Shorten long labels so charts stay readable."""
    text = str(text)
    return text if len(text) <= length else text[: length - 1] + "…"


# =====================================================================
#  The analysis (same steps as the desktop version)
# =====================================================================
def make_cards(values):
    """Build the 8 summary boxes as HTML."""
    html = '<div class="cards">'
    for name, value in values.items():
        html += f'<div class="card"><div class="value">{value}</div><div class="label">{name}</div></div>'
    return html + "</div>"


def build_report(source_name, df, numeric_cols, categorical_cols):
    """Create the final report text (shown on screen and saved as a file)."""
    rows, columns = df.shape
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    duplicates = int(df.duplicated().sum())
    empty_cols = [c for c in df.columns if df[c].isnull().all()]

    lines = ["FINAL REPORT", "=" * 40]
    lines.append(f"File:                {source_name}")
    lines.append(f"Rows:                {rows:,}")
    lines.append(f"Columns:             {columns:,}")
    lines.append(f"Missing values:      {total_missing:,}")
    lines.append(f"Duplicate rows:      {duplicates:,}")
    lines.append(f"Numerical columns:   {len(numeric_cols)}")
    lines.append(f"Categorical columns: {len(categorical_cols)}")
    lines.append(f"Empty columns:       {len(empty_cols)}")
    lines.append("")

    lines += ["MISSING VALUES", "-" * 40]
    if total_missing == 0:
        lines.append("No missing values.")
    else:
        for col, count in missing.items():
            if count > 0:
                lines.append(f"{col}: {count:,} missing ({count / rows * 100:.2f}%)")
    lines.append("")

    lines += ["COMPLETELY EMPTY COLUMNS", "-" * 40]
    if empty_cols:
        lines += [f"{col} is completely empty." for col in empty_cols]
    else:
        lines.append("No completely empty columns found.")
    lines.append("")

    lines += ["DUPLICATES", "-" * 40]
    lines.append("No duplicated rows." if duplicates == 0 else f"{duplicates:,} duplicated rows found.")
    return "\n".join(lines)


def analyse(link, uploaded_file):
    """Runs when the user presses Analyse. Returns everything the tabs need."""
    source = uploaded_file or link              # an uploaded file wins over a link
    if not source or not source.strip():
        raise gr.Error("Paste a dataset link or upload a CSV file first.")

    try:
        df = read_dataset(source)
    except Exception as error:
        raise gr.Error(f"Could not load the dataset: {error}")

    rows, columns = df.shape
    numeric_cols, categorical_cols = split_columns(df)
    source_name = os.path.basename(source.split("?")[0]) or source

    missing = df.isnull().sum()
    empty_cols = [c for c in df.columns if df[c].isnull().all()]
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    # ---- summary cards ----
    cards = make_cards({
        "Rows": f"{rows:,}",
        "Columns": f"{columns:,}",
        "Missing cells": f"{int(missing.sum()):,}",
        "Duplicate rows": f"{int(df.duplicated().sum()):,}",
        "Numerical columns": str(len(numeric_cols)),
        "Categorical columns": str(len(categorical_cols)),
        "Empty columns": str(len(empty_cols)),
        "Memory (MB)": f"{memory_mb:.2f}",
    })

    # ---- written report + a .txt file to download ----
    report = build_report(source_name, df, numeric_cols, categorical_cols)
    report_path = os.path.join(tempfile.mkdtemp(), "csv_report.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    # ---- columns table: names, types, missing, unique ----
    column_rows = []
    for number, col in enumerate(df.columns, start=1):
        percent = missing[col] / rows * 100 if rows else 0
        column_rows.append([number, col, str(df[col].dtype), int(missing[col]),
                            f"{percent:.2f}%", int(df[col].nunique())])
    columns_table = pd.DataFrame(
        column_rows, columns=["#", "Column", "Data type", "Missing", "Missing %", "Unique values"])

    # ---- numerical statistics ----
    if numeric_cols:
        stats_table = df[numeric_cols].describe().T.round(3)
        stats_table.insert(0, "Column", stats_table.index)
        stats_table = stats_table.reset_index(drop=True)
    else:
        stats_table = pd.DataFrame({"Info": ["No numerical columns found."]})

    # ---- categorical value counts (top 10 per column) ----
    cat_rows = []
    for col in categorical_cols:
        counts = df[col].value_counts().head(10)
        total = df[col].notna().sum()
        for value, count in counts.items():
            cat_rows.append([col, str(value), int(count), f"{count / total * 100:.1f}%"])
    if cat_rows:
        cat_table = pd.DataFrame(cat_rows, columns=["Column", "Value", "Count", "Share"])
    else:
        cat_table = pd.DataFrame({"Info": ["No categorical columns found."]})

    # ---- preview (first 200 rows, missing cells shown as empty) ----
    preview = df.head(200).astype(object)
    preview = preview.where(preview.notna(), "")

    status = f"Loaded **{source_name}** : {rows:,} rows × {columns:,} columns"

    # The order here must match the "outputs" list further down.
    return (df, status, cards, report, report_path, columns_table, stats_table,
            cat_table, preview, CHART_TYPES[0])


# =====================================================================
#  Charts
# =====================================================================
def new_figure():
    """An empty chart canvas with the dark background."""
    return Figure(figsize=(9, 5.5), facecolor=BLACK)


def message_figure(text):
    """A chart that only shows a message (used when a chart is not possible)."""
    fig = new_figure()
    fig.text(0.5, 0.5, text, ha="center", va="center", color=BUTTER,
             fontsize=13, fontfamily="serif")
    return fig


def style_axes(ax, title):
    """Give a chart the burgundy / butter / coffee look."""
    ax.set_facecolor(BLACK)
    ax.set_title(title, color=BUTTER, fontsize=13, fontfamily="serif", pad=12)
    ax.tick_params(colors=BUTTER_DIM)
    ax.xaxis.label.set_color(BUTTER_DIM)
    ax.yaxis.label.set_color(BUTTER_DIM)
    for spine in ax.spines.values():
        spine.set_color(COFFEE)


def rotate_labels(ax):
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def chart_missing_heatmap(df):
    fig = new_figure()
    # very big datasets: draw a random sample of 1000 rows
    sample = df if len(df) <= 1000 else df.sample(1000, random_state=1).sort_index()
    ax = fig.add_subplot(111)
    sns.heatmap(sample.isnull(), cbar=False, cmap=MISSING_CMAP, yticklabels=False, ax=ax)
    title = "Missing values map (butter = missing)"
    if len(df) > 1000:
        title += "  -  1000-row sample"
    style_axes(ax, title)
    rotate_labels(ax)
    fig.tight_layout()
    return fig


def chart_missing_bars(df):
    percent = df.isnull().mean() * 100
    percent = percent[percent > 0].sort_values().tail(30)
    if percent.empty:
        return message_figure("No missing values. Nice and clean!")
    fig = new_figure()
    ax = fig.add_subplot(111)
    ax.barh([short(c) for c in percent.index], percent.values, color=BURGUNDY_LIGHT)
    ax.set_xlabel("% missing")
    style_axes(ax, "Missing values per column")
    fig.tight_layout()
    return fig


def chart_histogram(df, numeric_cols, col):
    if not numeric_cols:
        return message_figure("No numerical columns in this dataset.")
    values = df[col].dropna()
    if values.empty:
        return message_figure(f"'{col}' has no data to plot.")

    fig = new_figure()
    # histogram on the left
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(values, bins=30, color=BURGUNDY_LIGHT, edgecolor=BLACK)
    ax1.set_xlabel(str(col))
    ax1.set_ylabel("Count")
    style_axes(ax1, f"Histogram of {short(col, 25)}")

    # boxplot on the right (shows outliers as dots)
    ax2 = fig.add_subplot(1, 2, 2)
    box = ax2.boxplot(values, patch_artist=True)
    box["boxes"][0].set(facecolor=BURGUNDY_LIGHT, edgecolor=BUTTER)
    for line in box["whiskers"] + box["caps"]:
        line.set_color(BUTTER_DIM)
    for line in box["medians"]:
        line.set_color(BUTTER)
    for dot in box["fliers"]:
        dot.set(markerfacecolor=BUTTER, markeredgecolor=BUTTER, markersize=3)
    ax2.set_xticks([])
    style_axes(ax2, "Boxplot (outliers)")
    fig.tight_layout()
    return fig


def chart_top_categories(df, categorical_cols, col):
    if not categorical_cols:
        return message_figure("No categorical columns in this dataset.")
    counts = df[col].value_counts().head(10)
    if counts.empty:
        return message_figure(f"'{col}' has no data to plot.")
    fig = new_figure()
    ax = fig.add_subplot(111)
    ax.bar(range(len(counts)), counts.values, color=BURGUNDY_LIGHT)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([short(v) for v in counts.index])
    ax.set_ylabel("Count")
    style_axes(ax, f"Top 10 values in {short(col, 30)}")
    rotate_labels(ax)
    fig.tight_layout()
    return fig


def chart_correlation(df, numeric_cols):
    if len(numeric_cols) < 2:
        return message_figure("Need at least 2 numerical columns\nfor a correlation map.")
    corr = df[numeric_cols].corr()
    fig = new_figure()
    ax = fig.add_subplot(111)
    show_numbers = len(numeric_cols) <= 12        # too many numbers get messy
    sns.heatmap(corr, annot=show_numbers, fmt=".2f", cmap=CORR_CMAP, vmin=-1, vmax=1,
                linewidths=0.5, linecolor=BLACK, ax=ax)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(colors=BUTTER_DIM)
    style_axes(ax, "Correlation between numerical columns")
    rotate_labels(ax)
    fig.tight_layout()
    return fig


def chart_scatter(df, numeric_cols, x_col, y_col):
    if not numeric_cols:
        return message_figure("No numerical columns in this dataset.")
    x, y = df[x_col], df[y_col]
    keep = x.notna() & y.notna()                  # drop rows with gaps
    x, y = x[keep], y[keep]
    if len(x) > 5000:                             # keep the chart quick
        pick = np.random.RandomState(1).choice(len(x), 5000, replace=False)
        x, y = x.iloc[pick], y.iloc[pick]
    fig = new_figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, color=BUTTER, alpha=0.6, s=14, edgecolors="none")
    ax.set_xlabel(str(x_col))
    ax.set_ylabel(str(y_col))
    style_axes(ax, f"{short(x_col, 25)} vs {short(y_col, 25)}")
    fig.tight_layout()
    return fig


def update_chart_options(df, kind):
    """Show the right column dropdowns for the chosen chart."""
    hidden = gr.Dropdown(choices=[], value=None, visible=False)
    if df is None:
        return hidden, hidden

    numeric_cols, categorical_cols = split_columns(df)

    if kind == "Histogram + boxplot":
        first = numeric_cols[0] if numeric_cols else None
        return gr.Dropdown(choices=numeric_cols, value=first, label="Column", visible=True), hidden

    if kind == "Top 10 categories":
        first = categorical_cols[0] if categorical_cols else None
        return gr.Dropdown(choices=categorical_cols, value=first, label="Column", visible=True), hidden

    if kind == "Scatter plot":
        first = numeric_cols[0] if numeric_cols else None
        second = numeric_cols[1] if len(numeric_cols) > 1 else first   # start with two different columns
        return (gr.Dropdown(choices=numeric_cols, value=first, label="X axis", visible=True),
                gr.Dropdown(choices=numeric_cols, value=second, label="Y axis", visible=True))

    return hidden, hidden        # the other charts need no column


def draw_chart(df, kind, col1, col2):
    """Draw the chosen chart."""
    if df is None:
        return None
    numeric_cols, categorical_cols = split_columns(df)
    try:
        if kind == "Missing values heatmap":
            return chart_missing_heatmap(df)
        if kind == "Missing values per column":
            return chart_missing_bars(df)
        if kind == "Histogram + boxplot":
            return chart_histogram(df, numeric_cols, col1)
        if kind == "Top 10 categories":
            return chart_top_categories(df, categorical_cols, col1)
        if kind == "Correlation heatmap":
            return chart_correlation(df, numeric_cols)
        if kind == "Scatter plot":
            return chart_scatter(df, numeric_cols, col1, col2)
    except Exception as error:
        return message_figure(f"Could not draw this chart:\n{error}")


# =====================================================================
#  Optional: AI insights with Google Gemini
# =====================================================================
def ai_insights(df):
    """Ask Gemini to explain the dataset. Only summary numbers are sent, not every row."""
    if df is None:
        raise gr.Error("Analyse a dataset first.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ("AI insights are switched off because no API key was found.\n\n"
                "To switch them on: get a free key at aistudio.google.com, then in your Space open "
                "**Settings > Variables and secrets** and add a **secret** named `GEMINI_API_KEY`.")

    # Build a short summary of the data for the prompt.
    numeric_cols, categorical_cols = split_columns(df)
    parts = [f"Rows: {len(df)}, columns: {df.shape[1]}, duplicate rows: {int(df.duplicated().sum())}"]
    missing = (df.isnull().mean() * 100).round(1)
    parts.append("Missing % per column:\n" + missing[missing > 0].to_string())
    parts.append("Column types:\n" + df.dtypes.astype(str).to_string())
    if numeric_cols:
        parts.append("Numerical statistics:\n" + df[numeric_cols[:20]].describe().T.round(2).to_string())
    for col in categorical_cols[:10]:
        parts.append(f"Top values of {col}: " + str(df[col].value_counts().head(5).to_dict()))
    summary = "\n\n".join(parts)[:8000]            # keep the prompt a sensible size

    prompt = (
        "You are a data analyst. Below is a summary of a CSV dataset.\n"
        "Write 5 to 7 short, specific insights for a beginner: data quality problems, "
        "interesting patterns, outliers, relationships between columns, and what to check next. "
        "Put each insight on its own line starting with '- '. No headings and no extra text.\n\n"
        + summary
    )

    try:
        from google import genai                    # imported here so the app starts without a key
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text
    except Exception as error:
        return f"The AI request failed: {error}"


# =====================================================================
#  The web page
# =====================================================================
with gr.Blocks(title="CSV Explorer") as demo:
    gr.HTML('<div class="app-title"><h1>CSV Explorer</h1>'
            "<p>paste a dataset link or upload a file, and see what's inside</p></div>")

    df_state = gr.State(None)       # remembers the loaded dataset for this visitor

    # ---- input row ----
    with gr.Row():
        link_box = gr.Textbox(label="Dataset link", placeholder="https://.../data.csv", scale=3)
        file_box = gr.File(label="...or upload a CSV file", file_types=[".csv"],
                           type="filepath", scale=2)
    gr.Examples(
        examples=[["https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"]],
        inputs=link_box, label="Try an example link")
    analyse_btn = gr.Button("Analyse", variant="primary")
    status_box = gr.Markdown()

    # ---- tabs ----
    with gr.Tabs():
        with gr.Tab("Overview"):
            cards_html = gr.HTML()
            report_box = gr.Textbox(label="Report", lines=20, interactive=False)
            report_file = gr.File(label="Download report (.txt)")

        with gr.Tab("Columns"):
            columns_table = gr.Dataframe(interactive=False)

        with gr.Tab("Statistics"):
            stats_table = gr.Dataframe(interactive=False)

        with gr.Tab("Categories"):
            cat_table = gr.Dataframe(interactive=False)

        with gr.Tab("Preview"):
            preview_table = gr.Dataframe(interactive=False)

        with gr.Tab("Charts"):
            with gr.Row():
                chart_type = gr.Dropdown(CHART_TYPES, value=CHART_TYPES[0], label="Chart")
                col1_box = gr.Dropdown(choices=[], visible=False, label="Column")
                col2_box = gr.Dropdown(choices=[], visible=False, label="Y axis")
            chart_plot = gr.Plot(label="Chart")

        with gr.Tab("AI insights (optional)"):
            gr.Markdown("Uses the Google Gemini API. Only summary numbers are sent, not your rows.")
            ai_btn = gr.Button("Write insights with Gemini")
            ai_output = gr.Markdown()

    # ---- wiring: what happens when the user does something ----
    chart_inputs = [df_state, chart_type, col1_box, col2_box]

    analyse_btn.click(
        analyse,
        inputs=[link_box, file_box],
        # same order as the tuple returned by analyse()
        outputs=[df_state, status_box, cards_html, report_box, report_file,
                 columns_table, stats_table, cat_table, preview_table, chart_type],
    ).then(
        update_chart_options, inputs=[df_state, chart_type], outputs=[col1_box, col2_box]
    ).then(
        draw_chart, inputs=chart_inputs, outputs=chart_plot
    )

    # choosing a different chart: refresh the dropdowns, then redraw
    chart_type.input(
        update_chart_options, inputs=[df_state, chart_type], outputs=[col1_box, col2_box]
    ).then(draw_chart, inputs=chart_inputs, outputs=chart_plot)

    # choosing a different column: just redraw
    col1_box.input(draw_chart, inputs=chart_inputs, outputs=chart_plot)
    col2_box.input(draw_chart, inputs=chart_inputs, outputs=chart_plot)

    ai_btn.click(ai_insights, inputs=df_state, outputs=ai_output)


if __name__ == "__main__":
    demo.launch(css=CSS)
