from dataset import Dataset

from dimensionality_reduction.pca_reduction import PCAReduction
from dimensionality_reduction.sammon_mapping_reduction import SammonMappingReduction
from dimensionality_reduction.tsne_reduction import TSNEReduction

from quality_measure_dimensionality_reduction.distance_correlation_measure import DistanceCorrelationMeasure
from quality_measure_dimensionality_reduction.neighborhood_preservation_measure import NeighborhoodPreservationMeasure
from quality_measure_dimensionality_reduction.trustworthiness_measure import TrustworthinessMeasure


datasets = ["data/artificial_dataset.csv", "data/magic04.csv", "data/winequality-red.csv"]
reductions = [PCAReduction, SammonMappingReduction, TSNEReduction]
measures = [DistanceCorrelationMeasure, NeighborhoodPreservationMeasure, TrustworthinessMeasure]

for data_path in datasets:
    print("Dataset:", data_path)
    dataset = Dataset(data_path)
    for reduction in reductions:
        print("\tReduction:", reduction.__name__)
        reduced_data = reduction(dataset).reduce()
        for measure in measures:
            quality = measure(dataset.getData(), reduced_data).evaluate()
            print("\t\tMeasure:", measure.__name__ + ", Quality:", quality)
