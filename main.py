from dataset import Dataset

# Import the new modular DR classes
from dimensionality_reduction.pca_reduction import PCAReduction
from dimensionality_reduction.tsne_reduction import TSNEReduction
from dimensionality_reduction.sammon_mapping_reduction import SammonMappingReduction

# Import the new modular quality measure classes
from quality_measure_dimensionality_reduction.trustworthiness_measure import TrustworthinessMeasure
from quality_measure_dimensionality_reduction.distance_correlation_measure import DistanceCorrelationMeasure
from quality_measure_dimensionality_reduction.neighborhood_preservation_measure import NeighborhoodPreservationMeasure


# Load dataset
data = Dataset("artificial_dataset.csv")
original_data = data.getData()

# Apply dimensionality reduction techniques
pca = PCAReduction(data, n_components=2)
tsne = TSNEReduction(data, n_components=2, perplexity=10)
sammon = SammonMappingReduction(data, n_components=2)

pca_reduced = pca.reduce()
tsne_reduced = tsne.reduce()
sammon_reduced = sammon.reduce()

# Store results in a dictionary
dr_results = {
    'PCA': pca_reduced,
    't-SNE': tsne_reduced,
    'Sammon Mapping': sammon_reduced
}

# Quality measures
print("=" * 70)
print("Dimensionality Reduction Quality Evaluation")
print("=" * 70)
print()

results_table = []

for dr_name, reduced_data in dr_results.items():
    print(f"\n{dr_name} Results:")
    print("-" * 50)

    # Instantiate each quality measure
    trust = TrustworthinessMeasure(original_data, reduced_data, n_neighbors=5)
    dist_corr = DistanceCorrelationMeasure(original_data, reduced_data)
    neigh_pres = NeighborhoodPreservationMeasure(original_data, reduced_data, n_neighbors=5)

    # Evaluate them
    trustworthiness_score = trust.evaluate()
    distance_corr_score = dist_corr.evaluate()
    neighborhood_pres_score = neigh_pres.evaluate()

    # Print results
    print(f"  Trustworthiness:           {trustworthiness_score:.4f}")
    print(f"  Distance Correlation:      {distance_corr_score:.4f}")
    print(f"  Neighborhood Preservation: {neighborhood_pres_score:.4f}")

    results_table.append({
        'Method': dr_name,
        'Trustworthiness': trustworthiness_score,
        'Distance Correlation': distance_corr_score,
        'Neighborhood Preservation': neighborhood_pres_score
    })

# Summary table
print("\n" + "=" * 70)
print("Summary Table")
print("=" * 70)
print(f"{'Method':<10} {'Trustworthiness':<20} {'Dist. Correlation':<20} {'Neighb. Preservation':<20}")
print("-" * 70)

for result in results_table:
    print(f"{result['Method']:<10} "
          f"{result['Trustworthiness']:<20.4f} "
          f"{result['Distance Correlation']:<20.4f} "
          f"{result['Neighborhood Preservation']:<20.4f}")

print("=" * 70)

# Find best method for each metric
print("\nBest performing methods:")
print("-" * 50)

metrics = ['Trustworthiness', 'Distance Correlation', 'Neighborhood Preservation']
for metric in metrics:
    best_method = max(results_table, key=lambda x: x[metric])
    print(f"  {metric}: {best_method['Method']} ({best_method[metric]:.4f})")

print("\n=== Quality Measures ===")
print(f"Silhouette Score: {sil:.4f}")
print(f"Calinski-Harabasz Index: {ch:.4f}")
print(f"Davies-Bouldin Index: {db:.4f}")