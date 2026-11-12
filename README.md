# Leukemia screening dashboard (decision support prototype)

## 1. Put the Kaggle outputs here
    models/leukemia_effnet.keras
    models/model_config.json
    data/test_predictions.csv      (sample data for building Power BI)

## 2. Run (PowerShell, venv active, from this folder)
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

Every analyzed image is saved to `scans.db` and mirrored to `exports/scans.csv`.

## 3. Power BI, step by step
Stage A (day one, no server needed)
 1. Power BI Desktop -> Get data -> Text/CSV -> exports/scans.csv
 2. Build visuals: cards (total scans, % flagged), donut (prediction), line (scans per day by scanned_at),
    bar (by lab), histogram (age), slicers (gender, lab, review_status).
 3. Click Refresh in Desktop to see new scans.

Stage B (company use)
 1. Move the log to SQL Server / PostgreSQL / Azure SQL: set DATABASE_URL (see src/db.py).
 2. In Power BI Desktop connect to that database, publish to the Power BI Service,
    and set scheduled refresh (or DirectQuery with automatic page refresh).
 3. Embed: Power BI Service -> File -> Embed report -> "Website or portal" (organization users sign in),
    copy the link, then start the app with it:
        $env:POWERBI_EMBED_URL = "https://app.powerbi.com/reportEmbed?..."
        python -m streamlit run app.py
    The report appears in the top-right panel. Do NOT use "Publish to web": it makes patient data public.
 4. For viewers without Power BI accounts you need Power BI Embedded with a backend token service.

## Notes
* Patient IDs only, never names. Add login and an encrypted database before real use.
* The OpenCV nucleus features are heuristic. Always check the green outline on the overlay.
* Validate on your own labelled images: a Kaggle-trained model will drop on a different microscope/stain.
* Clinical use needs regulatory approval and local validation.
## what the app does 
* Research prototype. Analyzes smear images similar to the training dataset and rejects other images with “Unable to analyze.” Results must be confirmed by a qualified hematologist and are not a diagnosis. Not validated on images from other labs or on patient-separated data.