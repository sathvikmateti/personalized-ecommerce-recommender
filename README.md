# Personalized E-Commerce Recommendation Engine

An end-to-end machine learning recommendation system built using Python, Scikit-learn, TensorFlow, FastAPI, MLflow, and Docker.

## Project Overview

This project uses customer interaction data to generate personalized product recommendations for e-commerce users.

The recommendation pipeline includes data preprocessing, exploratory data analysis, collaborative filtering, neural recommendation experiments, model evaluation, experiment tracking, and API deployment.

## Tech Stack

- **Programming:** Python, Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-learn, TensorFlow
- **Recommendation Models:** Item-Based Collaborative Filtering, Neural Collaborative Filtering (NCF)
- **Experiment Tracking:** MLflow
- **API:** FastAPI, Uvicorn
- **Containerization:** Docker

## Dataset

**Retailrocket E-Commerce Dataset**

Source: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

The dataset contains 2,756,101 customer interaction events, including:

- Product views
- Add-to-cart events
- Transactions

Implicit feedback weights are assigned as follows:

| Event | Weight |
|---|---:|
| View | 1 |
| Add to cart | 3 |
| Transaction | 5 |

Raw data is excluded from this repository because of its size.

## Recommendation Models

### Item-Based Collaborative Filtering

The production recommendation model uses item-based collaborative filtering with cosine similarity and nearest-neighbor search.

It generates recommendations by identifying products similar to those previously interacted with by a user.

### Neural Collaborative Filtering

TensorFlow-based neural collaborative filtering models were developed and evaluated as experimental alternatives.

The deployed API currently uses the item-based collaborative filtering model.

## Model Evaluation

A temporal holdout strategy was used to evaluate recommendation quality.

**Item-Based Collaborative Filtering — 1,871 eligible test users**

| Metric | Result |
|---|---:|
| Hit Rate@10 | 0.0930 |
| Precision@10 | 0.0093 |
| Recall@10 | 0.0930 |
| NDCG@10 | 0.0619 |

NCF experiments were also evaluated on a smaller 100-user diagnostic sample. Those results are not directly comparable to the full Item-CF evaluation.

MLflow is used for experiment tracking and recording model evaluation metrics.

## API

The recommendation engine is served through a FastAPI application.

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | API information |
| `GET /health` | Health check |
| `GET /recommend/{visitor_id}?n=10` | Personalized product recommendations |

### Example

```text
GET http://localhost:8000/recommend/1150086?n=10
```

The response contains the visitor ID, model name, recommendation count, and ranked product IDs with scores.

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure the trained model artifacts are available in the `models/` directory.

Start the API:

```bash
python -m uvicorn src.api.main:app --reload
```

## Docker Deployment

Build the Docker image:

```bash
docker build -t ecommerce-recommender-api:1.0 .
```

Run the container:

```bash
docker run -d --name ecommerce-recommender -p 8000:8000 ecommerce-recommender-api:1.0
```

Open:

```text
http://localhost:8000/docs
```

**Note:** Trained model artifacts are excluded from Git. They must be generated or supplied locally before building the Docker image.

## Future Improvements

- Automated training and retraining pipelines
- Cold-start recommendation strategies
- Additional ranking and evaluation experiments
- Cloud deployment and monitoring
- CI/CD automation

## Author

Sathvik Mateti

GitHub: https://github.com/sathvikmateti
