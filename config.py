# Физические параметры
WHEEL_RADIUS = 0.3   # метры
MAX_SPEED = 200      # км/ч
MIN_SPEED = 0        # км/ч

# Параметры для Random Forest
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'random_state': 42,
    'class_weight': 'balanced'
}