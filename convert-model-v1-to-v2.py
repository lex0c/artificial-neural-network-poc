from joblib import dump, load


def convertv1tov2(filepath):
    dump({"layers": load(filepath), "configs": {"loss": "mse"}}, filepath)


convertv1tov2("models/cartpole.joblib")
convertv1tov2("models/cartpole-checkpoint.joblib")
convertv1tov2("models/cartpole-finetuned.joblib")

