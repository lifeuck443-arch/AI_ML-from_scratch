import numpy as np

class DecisionStump:
    """
    Very simple weak learner: threshold on a single feature.
    It chooses feature, threshold and polarity to minimize weighted error.
    """
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1  # direction: 1 means predict +1 when x >= thresh, -1 otherwise

    def fit(self, X, y, sample_weight):
        n_samples, n_features = X.shape
        best_err = np.inf

        # iterate features and possible thresholds (midpoints between
        #  sorted values)
        for feature in range(n_features):
            feature_values = X[:, feature]
            sorted_idx = np.argsort(feature_values)
            sorted_vals = feature_values[sorted_idx]
            sorted_y = y[sorted_idx]
            sorted_w = sample_weight[sorted_idx]

            # candidate thresholds: midpoints between successive unique values,
            # plus below minimum and above maximum to consider all splits.
            unique_vals = np.unique(sorted_vals)
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
           
            thresholds = np.concatenate(([unique_vals[0] - 1e-8], thresholds, [unique_vals[-1] + 1e-8]))

            for thr in thresholds:
                # try opposite polarity: +1 if x < thr, else -1
               
                pred = np.ones(n_samples)
                pred[feature_values < thr] = -1
                err = np.sum(sample_weight[pred != y])

                if err < best_err:
                    best_err = err
                    self.feature_idx = feature
                    self.threshold = thr
                    self.polarity = 1

                # polarity = 1: predict +1 if x >= thr, else -1
                pred = np.ones(n_samples)
                pred[feature_values >= thr] = -1
                err = np.sum(sample_weight[pred != y])

                if err < best_err:
                    best_err = err
                    self.feature_idx = feature
                    self.threshold = thr
                    self.polarity = -1

        return self

    def predict(self, X):
        n = X.shape[0]
        pred = np.ones(n)
        feature_values = X[:, self.feature_idx]
        if self.polarity == 1:
            pred[feature_values < self.threshold] = -1
        else:
            pred[feature_values >= self.threshold] = -1
        return pred


class AdaBoost:
    """
    AdaBoost (SAMME-like) for binary classification with decision stumps.
    y must be in {-1, +1}.
    """
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0] # polarity = 1: predict +1 if x >= thr, else -1
        # initialize sample weights uniformly
        w = np.ones(n_samples) / n_samples

        self.learners = []
        self.alphas = []

        for m in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, w)
            pred = stump.predict(X)

            # weighted error
            err = np.sum(w[pred != y])

            # Avoid division by zero or error >= 0.5 edge cases:
            err = np.clip(err, 1e-10, 1 - 1e-10)

            # compute alpha (classifier weight)
            alpha = 0.5 * np.log((1 - err) / err)

            # update sample weights
            # w_i <- w_i * exp(-alpha * y_i * h_m(x_i))
            w *= np.exp(-alpha * y * pred)
            w /= np.sum(w)  # normalize

            # store
            self.learners.append(stump)
            self.alphas.append(alpha)

            # early stopping if perfect classification
            agg = self._aggregate_predictions(X)
            agg_sign = np.sign(agg)
            agg_sign[agg_sign == 0] = 1  # break ties in favor of +1
            accuracy = np.mean(agg_sign == y)
            # optional: break if perfect
            if accuracy == 1.0:
                # trim unused estimators
                break

        self.alphas = np.array(self.alphas)
        return self

    def _aggregate_predictions(self, X):
        # sum alpha_m * h_m(x)
        agg = np.zeros(X.shape[0])
        for alpha, learner in zip(self.alphas, self.learners):
            agg += alpha * learner.predict(X)
        return agg

    def predict(self, X):
        agg = self._aggregate_predictions(X)
        pred = np.sign(agg)
        pred[pred == 0] = 1
        return pred

    def predict_proba(self, X):
        # convert aggregated score to a "confidence-like" value in (0,1)
        agg = self._aggregate_predictions(X)
        # using sigmoid to map to (0,1)
        probs = 1.0 / (1.0 + np.exp(-2 * agg))  # steeper mapping
        # return probability of class +1 and -1 in columns
        return np.vstack([1 - probs, probs]).T


# ---------------------------
# Demo on synthetic data
# ---------------------------
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    # Create a simple 2D dataset of two Gaussian blobs
    n_pos = 150
    n_neg = 150
    pos = rng.normal(loc=[2, 2], scale=1.0, size=(n_pos, 2))
    neg = rng.normal(loc=[-2, -2], scale=1.0, size=(n_neg, 2))

    X = np.vstack([pos, neg])
    y = np.hstack([np.ones(n_pos), -np.ones(n_neg)])

    # shuffle
    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]

    # split train/test
    split = int(0.7 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # train AdaBoost
    model = AdaBoost(n_estimators=50)
    model.fit(X_train, y_train)

    # predict and evaluate
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"AdaBoost accuracy on test set: {acc * 100:.2f}%")

    # Example of predict_proba
    probs = model.predict_proba(X_test[:5])
    print("First 5 predicted probabilities (P(-1), P(+1)):\n", probs)
