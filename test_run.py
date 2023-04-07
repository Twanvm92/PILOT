import Runner as runner
import DataLoader as data_loader
import evaluate
from pathlib import Path

# runner.run(
#     "./datasets-input-neural-network/DatasetD2/Round1/", [4003, 10, 2], 100, 3.0, 0.75
# )

# runner.displayCM("./", 2)
# runner.displayROC("./", 2)

# binary classification tenfold cross validation for dataset D2 with C1 configuration
# runner.runExperiment(
#     "./datasets-input-neural-network/DatasetD2/", [4003, 10, 2], 100, 3.0, 0.75
# )

# multi-class classification tenfold cross validation for dataset D1 with C3 configuration
# runner.runExperiment(
#     "./datasets-input-neural-network/DatasetD1/", [5072, 10, 10, 10, 5], 100, 3.0, 0.75
# )

path_D1 = Path("./datasets-input-neural-network/DatasetD1/")
path_D2 = Path("./datasets-input-neural-network/DatasetD2/")

data_loader.combine_and_output_experiment_data(path_D1, "groundtruth")
data_loader.combine_and_output_experiment_data(path_D1, "prediction")
data_loader.combine_and_output_experiment_data(path_D2, "groundtruth")
data_loader.combine_and_output_experiment_data(path_D2, "prediction")

print(
    evaluate.evaluate_preds(
        path_D1 / "GroundTruth.csv", path_D1 / "Prediction.csv", "macro"
    )
)
print(evaluate.evaluate_preds(path_D2 / "GroundTruth.csv", path_D2 / "Prediction.csv"))

runner.displayCM(path_D2)
runner.displayCM(path_D1)
