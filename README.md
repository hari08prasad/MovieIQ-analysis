# MovieIQ: Predictive Analytics on Film Success

MovieIQ analyses 2,000 films and predicts whether a movie will be financially successful. A movie is **successful when `revenue > budget`**. The project covers data preparation, exploratory analysis, statistical testing, a Random Forest classifier and an interactive Streamlit dashboard.

**Live app:** [add your Streamlit link here]
**Full write-up:** [`MovieIQ_Report.docx`](MovieIQ_Report.docx)

## Key findings

- 80.7% of films in the dataset are successful (1,614 of 2,000), so the classes are imbalanced.
- Budget and revenue are strongly correlated (0.76). The other features are almost uncorrelated with success.
- T-tests: only `popularity` differs significantly between successful and unsuccessful films (p = 0.0397), and the effect is small. `runtime` (p = 0.127) and `vote_average` (p = 0.294) do not.
- Chi-Square: genre is not associated with success (p = 0.9946).
- Random Forest: test accuracy 0.738, below the 0.807 "always predict success" baseline. ROC AUC is 0.518 (about random), and the model finds only 9 of 77 failures.
- Conclusion: the available features carry almost no signal about success, and the dataset appears to be synthetic. The dashboard's prediction tab is a demo of the workflow, not business advice.

## Features of the dashboard

- Sidebar filters for genre and minimum vote average that update every chart and test
- Budget vs. revenue scatter plot, success rate by genre, boxplots and a correlation heatmap
- T-tests and Chi-Square test recomputed on the filtered data
- A prediction form that returns a success / not-success result from the trained model

## Project structure

```
MovieIQ/
├── MovieIQ.py            # Streamlit app
├── analysis.ipynb        # Stages 1-4: cleaning, EDA, tests, modelling
├── movies.csv            # Dataset
├── model.pkl             # Trained Random Forest
├── model_columns.pkl     # Feature order used by the model
├── requirements.txt      # Python dependencies
├── assets/               # Saved charts
├── MovieIQ_Report.docx   # Project report
└── README.md
```

## Setup and run

```bash
git clone https://github.com/hari08prasad/MovieIQ-analysis.git
cd MovieIQ-analysis
python -m venv venv
venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run MovieIQ.py
```

The app opens at http://localhost:8501.

## Method summary

| Stage | What was done |
|---|---|
| Data preparation | Parsed the `genres` strings, labelled 181 genre-less films "Unknown", created the `success` target |
| EDA | Scatter plot, genre trends, feature distributions, correlation heatmap |
| Statistical tests | Welch's T-tests on popularity, runtime and vote average; Chi-Square on genre |
| Modelling | One-hot genres, 80/20 stratified split, Random Forest (200 trees, balanced class weights) |
| Evaluation | Accuracy, precision, recall, confusion matrix, ROC AUC, permutation importance |
| App | Streamlit dashboard with filters, charts, tests and a prediction form |

`revenue` is excluded from the model's features because it defines the target (data leakage), and `title` is excluded because it is only an identifier.

## Limitations

- The data looks synthetic: features are nearly independent of each other and of the outcome.
- Success ignores marketing and distribution costs.
- Real drivers such as cast, director, franchise and release date are not in the data.

## Tech stack

Python, pandas, NumPy, scikit-learn, SciPy, seaborn, matplotlib, Streamlit, joblib
