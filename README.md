# Letterboxd Scraper

This is a set of Python scripts used to scrape and store [Letterboxd](https://letterboxd.com) user review data. It supports scheduled and manual scraping and stores results in MongoDB.

---

## GitHub Actions

This project includes a preconfigured GitHub Actions workflow that automatically triggers Python scripts to scrape LetterBoxd, compute user and film stats, and train a movie rating prediction model, and compute those predictions (see `full_process.yml`). This GitHub Action is scheduled to run automatically and can also be triggered manually through the GitHub GUI. The schedule is defined using cron syntax in the workflow file:

```yaml
schedule:
    - cron: "0 8 * * *" # runs every day at 8:00 AM UTC (2:00 AM CST / 3:00 AM CDT)
```

There are 4 separate GitHub Actions that can be triggered manually through the GitHub Action GUI. Each of these performs a portion of the full process:

1. Scraping LetterBoxd (`1_scrape.yml`, which runs `scrape/scraper.py`)
2. Computing user and film stats and superlatives (`2_stats.yml`, which runs `scrape/stats.py`)
3. Training a model to predict users' film ratings (`3_train.yml`, which runs `prediction/train_model.py`)
4. Using the model to make the predictions (`4_predict.yml`, which runs `prediction/predictor.py`)

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

4. Set environment variables.

5. Run the desired script:

```bash
python -m folder_name.script_name
```

---

## Project Structure

```bash
.github/
├──workflows/
    ├── full_process.yml     # Full scrape and computations action configuration
    ├── 1_scrape.yml           # Scrape action configuration
	├── 2_stats.yml	         # Compute stats action configuration
	├── 3_train.yml            # Train prediction model action configuration
	├── 4_predict.yml          # Compute predictions action configuration
├── CODEOWNERS		         # List of codeowners that must approve PR
scrape/
├── scraper.py		         # Scraping functionality
├── stats.py		         # Stats computation
prediction/
├── predictor.py	         # Model utilization
├── train_model.py	         # Model training
.env                         # Local environment variables (not in repository)
.gitignore
README.md
requirements.txt
```
