import pandas as pd

# ---------------- CLASSIFIERS ----------------
from capymoa.classifier import (
    HoeffdingTree,
    AdaptiveRandomForestClassifier,
    NaiveBayes,
    KNN
)

# ---------------- STREAM GENERATORS ----------------
from capymoa.stream.generator import (
    SEA,
    HyperPlaneClassification,
    RandomRBFGeneratorDrift
)

# ---------------------------------------------------
# Prequential Evaluation
# ---------------------------------------------------
def run_stream(stream, stream_name, n_samples=8000):

    results = []

    stream.restart()
    first_instance = stream.next_instance()

    models = {
        "HoeffdingTree": HoeffdingTree(schema=stream.schema),
        "AdaptiveRF": AdaptiveRandomForestClassifier(schema=stream.schema),
        "NaiveBayes": NaiveBayes(schema=stream.schema),
        "KNN": KNN(schema=stream.schema)
    }

    stream.restart()

    for model_name, model in models.items():

        correct = 0

        for i in range(n_samples):
            instance = stream.next_instance()
            y = instance.y_index

            pred = model.predict(instance)

            # train after prediction (prequential evaluation)
            model.train(instance)

            # only evaluate if prediction exists
            if pred is not None:
                pred = int(pred)
                if pred == y:
                    correct += 1


            acc = correct/max(1, i+1)


            results.append({
                "instance": i,
                "accuracy": acc,
                "model": model_name,
                "stream": stream_name
            })

        print(f"{stream_name} - {model_name} finished")

        stream.restart()

    return results


# ---------------------------------------------------
# Run All Experiments
# ---------------------------------------------------
all_results = []

print("\nRunning Abrupt Drift (SEA)...")
all_results += run_stream(SEA(), "abrupt_drift")

print("\nRunning Gradual Drift (Hyperplane)...")
all_results += run_stream(HyperPlaneClassification(), "gradual_drift")

print("\nRunning Recurring Drift (RBF Drift)...")
all_results += run_stream(RandomRBFGeneratorDrift(), "recurring_drift")

# Save results
df = pd.DataFrame(all_results)
df.to_csv("research_part1/results.csv", index=False)

print("\nExperiments completed successfully!")
