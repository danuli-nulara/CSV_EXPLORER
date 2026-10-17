# CSV Explorer

A small web app that explains a CSV file for you. Paste a dataset link or upload a file, and it shows an overview, column details, statistics, category counts, a data preview, and charts. It also has an optional AI insights tab powered by Google Gemini.

**Live demo:** [open on Hugging Face Spaces](https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME)

<!-- Add a screenshot: save it as screenshot.png in this folder, then remove these comment marks -->
<!-- ![CSV Explorer screenshot](screenshot.png) -->

## Features

- **Two ways to load data:** paste a link or upload a `.csv` file (up to 50 MB).
- **Share links are fixed automatically:** GitHub, Google Drive, Google Sheets and Dropbox links are converted to direct download links.
- **Overview:** rows, columns, missing cells, duplicate rows, empty columns, memory use, and a written report you can download as a `.txt` file.
- **Columns:** data type, missing values and unique values for every column.
- **Statistics:** count, mean, standard deviation, min, quartiles and max for numerical columns.
- **Categories:** the 10 most common values of each categorical column.
- **Preview:** the first 200 rows, with missing cells shown as empty.
- **Charts:** missing values heatmap, missing values per column, histogram with boxplot, top 10 categories, correlation heatmap, and scatter plot.
- **AI insights (optional):** Gemini writes a short plain-language summary. Only summary numbers are sent, never your rows.

## Built with

Python, [Gradio](https://www.gradio.app/), pandas, NumPy, Matplotlib, Seaborn, and the Google Gemini API (optional).

The colour theme uses burgundy, butter yellow, coffee and black.

## Run it on your computer

You need Python 3.10 or newer.

```bash
# 1. Get the code
git clone https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME

# 2. Install the packages
pip install gradio -r requirements.txt

# 3. Start the app
python app.py
```

Then open the local address shown in the terminal (usually http://127.0.0.1:7860).

Tip for Windows PowerShell: run the three commands one per line. PowerShell doesn't support `&&`.

## Turn on AI insights (optional)

The app works fine without this. To switch on the AI tab:

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Set it as an environment variable before starting the app:

   ```powershell
   # Windows PowerShell
   $env:GEMINI_API_KEY="your-key-here"
   python app.py
   ```

   ```bash
   # Mac / Linux
   export GEMINI_API_KEY="your-key-here"
   python app.py
   ```

If Google retires the default model, set `GEMINI_MODEL` to a current model name the same way.

Never commit your API key to GitHub.

## Project files

| File | What it does |
|------|--------------|
| `app.py` | The whole app: loading data, analysis, charts, AI insights, and the Gradio page |
| `requirements.txt` | The Python packages the app needs |
| `LICENSE` | MIT license |

## Notes

- Uploaded data is kept in the server's memory for each visitor and disappears when the app restarts.
- `app.py` includes a small do-nothing `@spaces.GPU` function. It exists only because free Hugging Face accounts must use ZeroGPU hardware for Gradio Spaces, and ZeroGPU refuses to start without one. It never runs, and the code does nothing special on your own computer.

## License

Released under the [MIT License](LICENSE).
