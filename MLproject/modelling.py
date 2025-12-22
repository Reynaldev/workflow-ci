import sys
import mlflow
import pandas as pd

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Workflow CI")
mlflow.sklearn.autolog()

if __name__ == "__main__":
    data = pd.read_csv("prdectid_preprocessing.csv")

    X_train, X_test, y_train, y_test = train_test_split(data['review_akhir'], data['Sentiment'], test_size=0.2, random_state=42)

    params = {
        "C": float(sys.argv[1] if len(sys.argv) > 1 else 1.0)
    }

    with mlflow.start_run():
        vectorizer = TfidfVectorizer(
            min_df=5,
            max_df=0.8,
            max_features=200
        )

        train_ds = vectorizer.fit_transform(X_train)
        test_ds = vectorizer.transform(X_test)

        model = svm.SVC(C=params["C"], kernel="linear")
        model.fit(train_ds, y_train)

        prediction = model.predict(test_ds)
        print(classification_report(y_test, prediction, target_names=["positive", "negative"]))
