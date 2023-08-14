import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(52)


def convert_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    else:
        return 2


if __name__ == '__main__':
    data = pd.read_csv('https://www.dropbox.com/s/4vu5j6ahk2j3ypk/titanic_train.csv?dl=1')

    data.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        axis=1,
        inplace=True
    )
    data.dropna(inplace=True)

    # Separate these back
    y = data['Survived'].astype(int)
    X = data.drop('Survived', axis=1)

    X['Sex'] = X['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    X['Embarked'] = X['Embarked'].apply(lambda x: convert_embarked(x))

    X_train, X_val, y_train, y_val = \
        train_test_split(X.values, y.values, stratify=y, train_size=0.8)

    # Make your code here...
    from sklearn import tree
    from sklearn.metrics import accuracy_score
    from tqdm import tqdm
    from time import sleep


    def create_bootstrap(ds, size=None):
        """Returns random indices from the dataset"""
        if not size:
            # If size is not defined returns max size
            return np.random.choice(range(ds.shape[0]), ds.shape[0])
        return np.random.choice(range(ds.shape[0]), size)


    class RandomForestClassifier():
        def __init__(self, n_trees=10, max_depth=np.iinfo(np.int64).max, min_error=1e-6):
            self.n_trees = n_trees
            self.max_depth = max_depth
            self.min_error = min_error
            self.forest = None
            self.is_fit = False

        def fit(self, X_train, y_train):
            self.forest = []

            # Your code for Step 3 here
            # tqdm creates a progress bar visualization
            for _ in tqdm(range(self.n_trees)):
                sleep(0.01)  # To see the progress bar
                model = tree.DecisionTreeClassifier(
                    max_features='sqrt',
                    max_depth=self.max_depth,
                    min_impurity_decrease=self.min_error, )
                indices = create_bootstrap(y_train)
                X_new = X_train[indices]
                y_new = y_train[indices]
                model.fit(X_new, y_new)
                self.forest.append(model)

            self.is_fit = True
            return self.forest

        def predict(self, nd_array):
            if not self.is_fit:
                raise AttributeError('The forest is not fit yet! Consider calling .fit() method.')

            # Your code for Step 4 here
            tree_predictions = np.empty(shape=(0, nd_array.shape[0]))
            for model in self.forest:
                # calculates predictions for every tree
                pred = model.predict(nd_array)
                tree_predictions = np.vstack((tree_predictions, pred))

            final_predictions = []
            for i in range(nd_array.shape[0]):
                # picks the class for the most popular votes
                predictions = tree_predictions[:, i]
                prediction = 1.0 if sum(predictions) / len(predictions) > 0.5 else 0.0
                final_predictions.append(prediction)

            return final_predictions

        @staticmethod
        def accuracy(real, predict):
            result = [x == y for x, y in zip(real, predict)]
            return sum(result) / len(result)


    def stage1():
        model = tree.DecisionTreeClassifier()
        model.fit(X_train, y_train)

        y_predict = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_predict)
        return round(accuracy, 3)


    def stage2():
        result = create_bootstrap(y_train, size=10)
        return result


    def stage3():
        c = RandomForestClassifier()
        models = c.fit(X_train, y_train)
        model = models[0]
        y_predict = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_predict)
        return round(accuracy, 3)


    def stage4():
        c = RandomForestClassifier()
        c.fit(X_train, y_train)
        result = c.predict(X_val)
        return result


    def stage5():
        c = RandomForestClassifier()
        c.fit(X_train, y_train)
        pred = c.predict(X_val)
        accuracy = c.accuracy(y_val, pred)
        return round(accuracy, 3)


    def stage6():
        accuracies = []
        for i in range(1, 21):
            c = RandomForestClassifier(n_trees=i)
            c.fit(X_train, y_train)
            pred = c.predict(X_val)
            accuracy = c.accuracy(y_val, pred)
            accuracies.append(round(accuracy, 3))
        return accuracies


    print(stage6()[:20])
