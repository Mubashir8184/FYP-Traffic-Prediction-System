# Traffic Flow Prediction System

Dashboard and backend for the FYP traffic prediction project. You get the dashboard (stats, charts, custom date/time prediction) and a Flask API that uses the trained model.

## How to run

1. Unzip the project and open terminal in that folder (same place where you see `index.html` and the `backend` folder).

2. Optional but good to do – create a virtual environment:
   - Windows: `python -m venv env` then `env\Scripts\activate`

3. Install dependencies:
   ```cmd
   pip install -r backend/requirements.txt
   ```

4. Start the server:
   ```cmd
   python backend/app.py
   ```

5. Open in browser: **http://127.0.0.1:5000**

If you have your trained model, put `rf_traffic_model.joblib` inside the `models` folder so the app uses it for predictions. If the file is missing it still runs with fallback data.

## What’s inside

- **Frontend:** `index.html` (main dashboard), `custom-prediction.html` (pick date/time and get prediction), `styles.css`, `dashboard.js`
- **Backend:** `backend/app.py` (Flask server + API), `backend/prediction.py` (loads model and predicts)
- **Data:** `data/processed/processed_full.csv` – the app reads from here for stats and charts
- **Models:** put your `.joblib` model in the `models` folder

That’s it. If something doesn’t run check that you’re in the project root and Python can find the packages (step 3).
