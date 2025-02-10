from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def rf_train_search(X_train, y_train):
    print('\nRandom Forest parameter search is selected.')
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],  # Number of trees
        'max_depth': [None, 10, 20, 30],      # Depth of trees
        'min_samples_split': [2, 5, 10],      # Minimum samples to split a node
        'min_samples_leaf': [1, 2, 4],        # Minimum samples per leaf
        'bootstrap': [True, False]            # Bootstrap sampling
    }

    # Initialize Grid Search
    rf_model = GridSearchCV(RandomForestClassifier(random_state=1), 
                            param_grid, 
                            n_jobs=-1,  # Use all available cores
                            cv=2)  # Show progress

    print('Random Forest Train...')
    rf_model.fit(X_train, y_train)
    print('Random Forest Train Finished.')

    return rf_model.best_params_, rf_model.best_estimator_

def rf_train(X_train, y_train):
    # Default Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, 
                                      min_samples_split=2, min_samples_leaf=1, 
                                      bootstrap=True, random_state=1)
    
    print('\nRandom Forest Train...')
    rf_model.fit(X_train, y_train)
    print('Random Forest Train Finished.')
    
    return rf_model
