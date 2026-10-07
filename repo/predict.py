"""Predict test.csv with the saved champion models and write submission.csv (id,target_popularity).
usage: python predict.py"""
import config as C
from src import inference

if __name__ == "__main__":
    champ = inference.load_champion()
    sub, P = inference.predict(champ)
    print("submission:", C.SUBMISSION_CSV, sub.shape, list(sub.columns))
    print("prediction distribution:\n", sub[C.TARGET].value_counts(normalize=True).sort_index().round(3))
    print(sub.head())
