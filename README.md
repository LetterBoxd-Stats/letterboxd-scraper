# Letterboxd Scraper

This is a set of Python scripts used to scrape and store [Letterboxd](https://letterboxd.com) user review data. It supports scheduled and manual scraping and stores results in MongoDB.

---

## Environment Variables

These variables are loaded via dotenv for local development and should also be added to your GitHub Action repository secrets.

| Secret Name                  | Description                                  |
| ---------------------------- | -------------------------------------------- |
| `DB_URI`                     | MongoDB connection URI                       |
| `DB_NAME`                    | MongoDB database name                        |
| `DB_USERS_COLLECTION`        | Collection name for user reviews             |
| `DB_FILMS_COLLECTION`        | Collection name for film metadata            |
| `DB_SUPERLATIVES_COLLECTION` | Collection name for superlatives             |
| `DB_MODELS_COLLECTION`       | Collection name for prediction models        |
| `LETTERBOXD_USERNAMES`       | Comma-separated list of usernames to scrape  |
| `LETTERBOXD_GENRES`          | Comma-separated list of genres in LetterBoxd |
| `ENV`                        | Environment (`prod` or `dev`)                |

---

## Local Development

1. Clone the repo

2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the desired script:

```bash
python -m folder_name.script_name
```

---

## Scraping and Stats

This project includes a preconfigured GitHub Actions workflow that automatically triggers the scraper, computes stats, and retrains the prediction models on a schedule (see `.github/workflows/scrape.yml`). This GitHub Action can also be triggered manually through the GitHub GUI. If you're setting this up in your own GitHub repo, ensure your GitHub repository secrets are configured.

The schedule is defined using cron syntax in the workflow file:

```yaml
schedule:
    - cron: "0 8 * * *" # runs every day at 8:00 AM UTC (2:00 AM CST / 3:00 AM CDT)
```

There is a separate preconfigured GitHub Actions workflow that triggers only a computation of the stats (see `.github/workflows/stats.yml`). This action is triggered manually through the GitHub GUI.

There is another separate preconfigured GitHub Actions workflow that triggers only the training of the models (see `.github/workflows/prediction.yml`). This action is triggered manually through the GitHub GUI

---

## Project Structure

```bash
.github/
├──workflows/
    ├── scrape.yml     # Scrape action configuration
	├── stats.yml	   # Compute stats action configuration
	├── prediction.yml # Train prediction model action configuration
├── CODEOWNERS		   # List of codeowners that must approve PR
scrape/
├── scraper.py		   # Scraping functionality
├── stats.py		   # Stats computation
prediction/
├── predictor.py	   # Model utilization
├── train_model.py	   # Model training
.env                   # Local environment variables (not in repository)
.gitignore
README.md
requirements.txt
```
