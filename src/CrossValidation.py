from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

from NetworkModel import *
from Balance import *

class CrossValidation:
    def cross_valid_and_fit_model(self, df, model, epochs):
        n_splits = 2
        n_times = 5
        acc_sum = 0
        f1_sum = 0

        X = df.drop(columns=['value'])
        Y = df['value'].values
        kf = StratifiedKFold(n_splits=n_splits, random_state=np.random, shuffle=True)
        kf.get_n_splits(X)

        for i in range(n_times):
            for train_index, test_index in kf.split(X,Y):
                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                X_train, y_train = balance_dataset(X_train, y_train)

                model_history = model.fit(  X_train,
                                            y_train,
                                            epochs=epochs,
                                            validation_data=(X_test, y_test),
                                            verbose=1)
                acc_sum += model_history.history['val_acc'][-1]
                f1_sum += model_history.history['val_f1_score'][-1]

        return acc_sum/(n_splits*n_times), f1_sum/(n_splits*n_times)
