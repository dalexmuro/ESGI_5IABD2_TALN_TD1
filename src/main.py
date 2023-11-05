import click
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from src.data.make_dataset import make_dataset
from src.features.make_features import make_features
from src.model.main import make_model
import joblib
import pickle
import gzip


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/model.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    if task == "find_comic_name":
        pass
    else:
        df = make_dataset(input_filename)
        X, y = make_features(df, task)

        model = make_model(task)
        model.fit(X, y)

        with gzip.open(model_dump_filename, 'wb') as f:
            pickle.dump(model, f)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/test.csv", help="File training data")
@click.option("--model_dump_filename", default="models/text_classification.pkl", help="File to load model from")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(task, input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)
    if task == "find_comic_name":
        # Predict is_comic and is_name
        X_is_comic = make_features(df, "is_comic_video", output_required=False)
        model_is_comic = pickle.load(gzip.open("models/is_comic_video.json", 'rb'))
        df["is_comic_video_prediction"] = model_is_comic.predict(X_is_comic)
        X_is_name = make_features(df, "is_name", output_required=False)
        model_is_name = pickle.load(gzip.open("models/is_name.json", 'rb'))
        preds_is_name = model_is_name.predict(X_is_name)
        df["is_name_prediction"] = ""
        cpt = 0
        for index, row in df.iterrows():
            df.at[index, "is_name_prediction"] = preds_is_name[cpt:cpt + len(row["tokens"])]
            cpt += len(row["tokens"])

        # Combining results
        y_pred = []
        for index, row in df.iterrows():
            if row["is_comic_video_prediction"] == 1:
                name = ""
                for i, pred in enumerate(row["is_name_prediction"]):
                    if pred == 1:
                        name += " " + row["tokens"][i]
                y_pred.append(name)
            else:
                y_pred.append("None")
        df[task + "_prediction"] = y_pred
    else:
        X = make_features(df, task, output_required=False)
        model = pickle.load(gzip.open(model_dump_filename, 'rb'))
        if task == "is_comic_video":
            df[task + "_prediction"] = model.predict(X)
        elif task == "is_name":
            df[task + "_prediction"] = ""
            preds = model.predict(X)
            cpt = 0
            for index, row in df.iterrows():
                df.at[index, task + "_prediction"] = preds[cpt:cpt + len(row["tokens"])]
                cpt += len(row["tokens"])
    return df.to_csv(output_filename, index=False)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    # Read CSV
    df = make_dataset(input_filename)
    if task == "find_comic_name":
        # Predict is_comic and is_name
        X_is_comic = make_features(df, "is_comic_video", output_required=False)
        model_is_comic = pickle.load(gzip.open("models/is_comic_video.json", 'rb'))
        df["is_comic_video_prediction"] = model_is_comic.predict(X_is_comic)
        X_is_name = make_features(df, "is_name", output_required=False)
        model_is_name = pickle.load(gzip.open("models/is_name.json", 'rb'))
        preds_is_name = model_is_name.predict(X_is_name)
        df["is_name_prediction"] = ""
        cpt = 0
        for index, row in df.iterrows():
            df.at[index, "is_name_prediction"] = preds_is_name[cpt:cpt + len(row["tokens"])]
            cpt += len(row["tokens"])

        # Combining results
        y_pred = []
        y = []
        for index, row in df.iterrows():
            if row["is_comic_video_prediction"] == 1:
                name = ""
                for i, pred in enumerate(row["is_name_prediction"]):
                    if pred == 1:
                        name += " " + row["tokens"][i]
                y_pred.append(name)
            else:
                y_pred.append("None")
            if row["is_comic"] == 1:
                name = ""
                for i, pred in enumerate(row["is_name"]):
                    if pred == 1:
                        name += " " + row["tokens"][i]
                y.append(name)
            else:
                y.append("None")


        print(f"Global accuracy {100 * round(accuracy_score(y, y_pred), 2)}%")
    else:
        # Make features (tokenization, lowercase, stopwords, stemming...)
        X, y = make_features(df, task)

        model = make_model(task)

        # Run k-fold cross validation. Print results
        return evaluate_model(model, task, X, y)

def evaluate_model(model, task, X, y):
    # Scikit learn has function for cross validation
    results = cross_validate(model, X, y, scoring="accuracy", cv=10, return_estimator=True)
    print(results)
    scores = results["test_score"]

    print(f"Accuracy per fold : {np.around(scores, 2)}")
    print(f"Global accuracy {100 * round(np.mean(scores), 2)}% + {round(np.std(scores), 2)}")


    if task == "is_comic_video":
        vocab = results['estimator'][0]['vectorizer'].vocabulary_
        keys = list(vocab.keys())

        importance = results['estimator'][0]['classifier'].feature_importances_
        features = pd.DataFrame(importance, index=keys, columns=['importance']).sort_values('importance', ascending=False)
        print("Top used features")
        print(features.head(10))

    return results


cli.add_command(train)
cli.add_command(test)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()