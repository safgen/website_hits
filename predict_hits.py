import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, DMatrix, cv, plot_importance


def train_function(model, X, y, X_t, y_t, max_iter=100, verbose=False):
    if model == 'MLP':
        Model = MLPRegressor(hidden_layer_sizes= (100,100,), max_iter=max_iter, verbose=verbose, tol=1e-4, batch_size=200,
                             early_stopping=True, validation_fraction=0.1)

    elif model == 'XGB':
        Model = XGBRegressor(verbose=verbose, max_depth=5, silent=True, max_iter=max_iter, reg_alpha=10,
                             n_estimators=200,
                             colsample_bytree=0.3, objective='reg:squarederror')
    elif model == 'SGD':
        Model = SGDRegressor(max_iter=300, verbose=verbose, tol=1e-4)
    else:
        print('Invalid Model choice')
        sys.exit(1)
    Model.fit(X, y)
    prediction = Model.predict(X_t)
    ms_error = mean_squared_error(y_t, prediction)
    print('MSE for ' + model + ' is = ' + str(ms_error))

    return Model, ms_error


if __name__ == "__main__":
    nRowsRead = None
    full_data = pd.read_csv('feature_engineered_data.csv', nrows=nRowsRead).drop(columns=['Unnamed: 0'])
    data = full_data[full_data["hits"].notna()]
    data_unknown = full_data[full_data["hits"].isna()]

    # test train split
    cols2drop = ['index', 'row_num', 'hits']
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=cols2drop), data['hits'].values,
                                                        test_size=0.25, random_state=52)
    # Training different models with configurations tuned and tested in jupyter notebook
    trained_sgd, mse_sgd = train_function(model='SGD', X=X_train, y=y_train, X_t=X_test, y_t=y_test)
    # for faster execution skip this MLPRegressor Model
    trained_mlp, mse_mlp = train_function(model='MLP', X=X_train, y=y_train, X_t=X_test, y_t=y_test)
    trained_xgb, mse_xgb = train_function(model='XGB', X=X_train, y=y_train, X_t=X_test, y_t=y_test)
    
    # selecting the best model
    if mse_xgb <= mse_mlp and mse_xgb <= mse_sgd:
        model = trained_xgb
        rmse = np.sqrt(mse_xgb)
    elif mse_mlp <= mse_xgb and mse_mlp <= mse_sgd:
        model = trained_mlp
        rmse = np.sqrt(mse_mlp)
    else:
        model = trained_sgd
        rmse = np.sqrt(mse_sgd)

    # model = trained_xgb
    # rmse = np.sqrt(mse_xgb)

    print('RMSE value for the best model is:' + str(rmse))

    # predictions on the unknown hits data and saving it to a csv file
    output_data = data_unknown.drop(columns=cols2drop)
    output_preds = model.predict(output_data)
    output_dict = dict(zip(data_unknown['row_num'].values, output_preds))
    output_df = pd.DataFrame(output_dict.items(), columns=['row_num', 'predicted_hits']).to_csv('output.csv',
                                                                                                index=False)

    plot_importance(trained_xgb)
    plt.show()
