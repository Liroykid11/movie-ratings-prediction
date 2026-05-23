import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv(r'C:\Users\Arseni\Documents\movie-ratings-prediction\movie-ratings-prediction\data\processed\cleaned_movies_v2.csv')

X = df.drop('Vote_Average', axis = 1)
y = df['Vote_Average']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=100)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Rating', fontsize=12)
plt.ylabel('Predicted Rating', fontsize=12)
plt.title(f'Linear Regression: Predictions vs Actual\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/models/linear_regression/linear_regression_predictions_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()

errors = y_test - predictions

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.title(f'Model Error Distribution\nMean Error: {errors.mean():.4f}, Std: {errors.std():.4f}', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/models/linear_regression/linear_regression_error_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# LassoCV


lasso_cv = LassoCV(
    alphas=np.logspace(-4, 1, 50),  
    cv=5,                          
    random_state=42,
    max_iter=10000,
    selection='random' 
)

lasso_cv.fit(X_train, y_train)

lasso_predictions = lasso_cv.predict(X_test)

coef_alpha_lasso = lasso_cv.alpha_ # 0.002682695795279727
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_predictions))
lasso_r2 = r2_score(y_test, lasso_predictions)


plt.figure(figsize=(10, 6))
plt.scatter(y_test, lasso_predictions, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Rating', fontsize=12)
plt.ylabel('Predicted Rating', fontsize=12)
plt.title(f'LassoCV: Predictions vs Actual\nR² = {lasso_r2:.4f}, RMSE = {lasso_rmse:.4f}', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/models/lassocv/lassocv_predictions_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()


plt.figure(figsize=(10, 6))
plt.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_, ':')
plt.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=-1), 'k', label='Average CV Error', linewidth=2)
plt.axvline(lasso_cv.alpha_, linestyle='--', color='r', label=f'Best alpha = {lasso_cv.alpha_:.6f}')
plt.xlabel('Alpha', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('LassoCV: Alpha Selection Path', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/models/lassocv/lassocvlassocv_alpha_path.png', dpi=150, bbox_inches='tight')
plt.close()


# Ridge

ridge_cv = RidgeCV(
    alphas=np.logspace(-4, 3, 50), 
    cv=5,                           
    scoring='r2',                  
)

ridge_cv.fit(X_train, y_train)
coef_alpha_ridge = ridge_cv.alpha_

ridge_predictions = ridge_cv.predict(X_test)

ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))
ridge_r2 = r2_score(y_test, ridge_predictions)


plt.figure(figsize=(10, 6))
plt.scatter(y_test, ridge_predictions, alpha=0.3, s=10, color='seagreen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Rating', fontsize=12)
plt.ylabel('Predicted Rating', fontsize=12)
plt.title(f'RidgeCV: Predictions vs Actual\nR² = {ridge_r2:.4f}, RMSE = {ridge_rmse:.4f}', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/models/ridgecv/ridgecv_predictions_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()

ridge_errors = y_test - ridge_predictions

plt.figure(figsize=(10, 6))
plt.scatter(ridge_predictions, ridge_errors, alpha=0.3, s=10, color='seagreen')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Rating', fontsize=12)
plt.ylabel('Error (Actual - Predicted)', fontsize=12)
plt.title('RidgeCV: Error vs Predicted Values', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/models/ridgecv/ridgecv_error_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()

#saving models

joblib.dump(model, 'models/linear_regression_model.pkl')
joblib.dump(lasso_cv, 'models/lassocv_model.pkl')
joblib.dump(ridge_cv, 'models/ridgecv_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

#compare models

metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'LassoCV', 'RidgeCV'],
    'R²': [r2, lasso_r2, ridge_r2],
    'RMSE': [rmse, lasso_rmse, ridge_rmse]
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bars1 = axes[0].bar(metrics_df['Model'], metrics_df['R²'], 
                    color=['steelblue', 'coral', 'seagreen'])
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('Model Comparison: R² Score', fontsize=14)
axes[0].set_ylim(0, max(metrics_df['R²']) + 0.05)
for bar, val in zip(bars1, metrics_df['R²']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontsize=11)

bars2 = axes[1].bar(metrics_df['Model'], metrics_df['RMSE'],
                    color=['steelblue', 'coral', 'seagreen'])
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_title('Model Comparison: RMSE', fontsize=14)
for bar, val in zip(bars2, metrics_df['RMSE']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('results/models/comparison/model_comparison_bars.png', dpi=150, bbox_inches='tight')
plt.close()