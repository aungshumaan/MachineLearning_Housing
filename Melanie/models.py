from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone


def modelfitRF(model, x_train, features, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    model.fit(x_train, y_train)
        
    #Predict training set:
    status = model.predict(x_validate)
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(model, x_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
    
    #Print model report:
    print("\nModel Report")
    print("Error : %f" % metrics.mean_squared_error(y_validate, status))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        print('The R^2 score is:',model.score(x_validate, y_validate))
        print('OOB_score is:', model.oob_score_)
        
    #Print Feature Importance
    if printFeatureImportance:
        feat_imp = pd.Series(model.feature_importances_, features).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
    
    #final predictions
    #final_status = np.expm1(model.predict(x_test))

    #submission = pd.DataFrame({'Id': test_ID, 'SalePrice':final_status})
    #submission.to_csv('rf.csv', index=False)


def modelfitxgb(model, x_train, useTrainCV=True, cv_folds=5, printFeatureImportance=True):
    
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train, label=y_train)
        xgtrain = xgb.DMatrix(np.matrix(x_train), label = y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'],
                          nfold=cv_folds, early_stopping_rounds=50, metrics = 'rmse')
        model.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    model.fit(x_train, y_train, eval_metric='rmse')
        
    #Predict training set:
    status = model.predict(x_validate)
    #plt.scatter(y_validate, status)  
    
    #Print model report:
    print ("\nModel Report")
    print ("Error : %f" % metrics.mean_squared_error(y_validate, status))
    print(model.score(x_validate, y_validate))  

    #final predictions
    #final_status = np.expm1(model.predict(x_test))

    #submission = pd.DataFrame({'Id': test_ID, 'SalePrice':final_status})
    #submission.to_csv('xgboost_fs.csv', index=False)

 def modelfitGB(model, x_train, features, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    model.fit(x_train, y_train)
        
    #Predict training set:
    status = model.predict(x_validate)
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(model, x_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
    
    #Print model report:
    print("\nModel Report")
    print("Error : %f" % metrics.mean_squared_error(y_validate, status))
    print(model.score(x_validate, y_validate))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(model.feature_importances_, features).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
    
    
    #final predictions
    #final_status = np.expm1(model.predict(x_test))

    #submission = pd.DataFrame({'Id': test_ID, 'SalePrice':final_status})
    #submission.to_csv('gdboost_fs.csv', index=False)


def modelfitKNN(model, x_train, performCV=True, cv_folds=5):
    #Fit the algorithm on the data
    model.fit(x_train, y_train)
        
    #Predict training set:
    status = model.predict(x_validate)
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(model, x_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
    
    #Print model report:
    print("\nModel Report")
    print("Error : %f" % metrics.mean_squared_error(y_validate, status))
    print(model.score(x_validate, y_validate))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    #if printFeatureImportance:
    #    feat_imp = pd.Series(model.feature_importances_, features).sort_values(ascending=False)
    #    feat_imp.plot(kind='bar', title='Feature Importances')
    #    plt.ylabel('Feature Importance Score')
    
    plt.scatter(y_validate, status)
    
    #final predictions
    #final_status = np.expm1(model.predict(x_test))

    ##submission = pd.DataFrame({'Id': test_ID, 'SalePrice':final_status})
    #submission.to_csv('gdboost_fs.csv', index=False)


def stacking_regression(models, meta_model, X_train, y_train, X_test,
             transform_target=None, transform_pred=None,
             metric=None, n_folds=3, average_fold=True,
             shuffle=False, random_state=0, verbose=1):
   
    if verbose > 0:
        print('metric: [%s]\n' % metric.__name__)

    # Split indices to get folds
    kf = KFold(n_splits = n_folds, shuffle = shuffle, random_state = random_state)

    if X_train.__class__.__name__ == "DataFrame":
    	X_train = X_train.as_matrix()
    	X_test = X_test.as_matrix()

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))

    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('model %d: [%s]' % (model_counter, model.__class__.__name__))

        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            
            # Clone the model because fit will mutate the model.
            instance = clone(model)
            
            # Fit 1-st level model
            instance.fit(X_tr, transformer(y_tr, func = transform_target))
            
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = transformer(instance.predict(X_te), func = transform_pred)
            
            # Predict full test set
            S_test_temp[:, fold_counter] = transformer(instance.predict(X_test), func = transform_pred)

            # Delete temperatory model
            del instance

            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, metric(y_te, S_train[te_index, model_counter])))

        # Compute mean or mode of predictions for test set
        if average_fold:
            S_test[:, model_counter] = np.mean(S_test_temp, axis = 1)
        else:
            model.fit(X_train, transformer(y_train, func = transform_target))
            S_test[:, model_counter] = transformer(model.predict(X_test), func = transform_pred)

        if verbose > 0:
            print('    ----')
            print('    MEAN:   [%.8f]\n' % (metric(y_train, S_train[:, model_counter])))

    # Fit our second layer meta model
    
    meta_model.fit(S_train, transformer(y_train, func = transform_target))
    
    # Make our final prediction
    stacking_prediction = transformer(meta_model.predict(S_test), func = transform_pred)

    return stacking_prediction