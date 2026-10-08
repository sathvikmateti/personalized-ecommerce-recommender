from src.models.recommender import ItemCFRecommender


print("Loading recommendation model...")

recommender = ItemCFRecommender()

print("Model loaded.")

visitor_id = 1150086

recommendations = recommender.recommend(
    visitor_id=visitor_id,
    n=10
)

print(f"\nRecommendations for customer {visitor_id}:")

for rank, recommendation in enumerate(
    recommendations,
    start=1
):
    print(
        rank,
        recommendation["itemid"],
        round(recommendation["score"], 4)
    )