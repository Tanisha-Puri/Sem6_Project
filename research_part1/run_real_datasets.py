import pandas as pd

from capymoa.classifier import (
    HoeffdingTree,
    AdaptiveRandomForestClassifier,
    NaiveBayes,
    KNN
)

from capymoa.stream import ARFFStream


# -------- helper to auto-detect target ----------
def load_stream(path):

    df = pd.read_csv(path, nrows=5)
    target_col = df.columns[-1]   # last column = label
    print(f"Loading {path} | target = {target_col}")

    return CSVStream(path, target=target_col), target_col


def evaluate_stream(stream, name, n_samples=15000):

    stream.restart()
    first_instance = stream.next_instance()

    models = {
        "HoeffdingTree": HoeffdingTree(schema=stream.schema),
        "AdaptiveRF": AdaptiveRandomForestClassifier(schema=stream.schema),
        "NaiveBayes": NaiveBayes(schema=stream.schema),
        "KNN": KNN(schema=stream.schema)
    }

    stream.restart()

    results = []

    for model_name, model in models.items():

        correct = 0

        for i in range(n_samples):

            if not stream.has_more_instances():
                break

            instance = stream.next_instance()
            y = instance.y_index

            pred = model.predict(instance)
            model.train(instance)

            if pred is not None and int(pred) == y:
                correct += 1

            acc = correct / max(1, i+1)

            results.append({
                "instance": i,
                "accuracy": acc,
                "model": model_name,
                "stream": name
            })

        print(name, model_name, "done")
        stream.restart()

    return results


# ---------------- RUN ----------------
all_results = []

for file, name in [
    ("research_part1/processed/electricity.arff", "electricity"),
    ("research_part1/processed/airlines.arff", "airlines")
]:
    stream = ARFFStream(file)
    all_results += evaluate_stream(stream, name)


pd.DataFrame(all_results).to_csv("research_part1/real_results.csv", index=False)

print("\nReal dataset experiments finished!")
