from fastapi import FastAPI, HTTPException, Query

from src.models.recommender import ItemCFRecommender


app = FastAPI(
    title="Personalized E-Commerce Recommendation API",
    description="API for personalized product recommendations.",
    version="1.0.0"
)

recommender = ItemCFRecommender()


@app.get("/")
def root():
    return {
        "message": "Personalized E-Commerce Recommendation API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Item-Based Collaborative Filtering",
        "customers": len(recommender.user_to_idx),
        "products": len(recommender.item_to_idx)
    }


@app.get("/recommend/{visitor_id}")
def recommend(
    visitor_id: int,
    n: int = Query(default=10, ge=1, le=50)
):
    recommendations = recommender.recommend(
        visitor_id=visitor_id,
        n=n
    )

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="Customer not found or no recommendations available."
        )

    return {
        "visitor_id": visitor_id,
        "model": "item_cf",
        "count": len(recommendations),
        "recommendations": recommendations
    }