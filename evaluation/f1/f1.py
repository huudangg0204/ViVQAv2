from sklearn.metrics import f1_score
import numpy as np

class F1_micro:
    def compute_score(self, gts, res):
        """
        Main function to compute F1 score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: f1 (float) : computed F1 score for the corpus
        """
        res = {key: value.split() for key, value in res.items()}
        scores = []
        for key in res:
            scores.append(f1_score(gts[key], res[key], average="micro", zero_division=0))

        scores = np.array(scores)

        return scores.mean(), scores

    def __str__(self) -> str:
        return "F1_micro"

class F1_macro:
    def compute_score(self, gts, res):
        """
        Main function to compute F1 score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: f1 (float) : computed F1 score for the corpus
        """
        res = {key: value.split() for key, value in res.items()}
        scores = []
        for key in res:
            scores.append(f1_score(gts[key], res[key], average="macro", zero_division=0))

        scores = np.array(scores)

        return scores.mean(), scores

    def __str__(self) -> str:
        return "F1_macro"