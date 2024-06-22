import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# 加载数据
data = pd.read_csv('D:/20224329/基于XGBoost的印度工科毕业生的薪资预测/dataset/Engineering_graduate_salary.csv')

# 处理DOB列，转换为年龄
data['DOB'] = pd.to_datetime(data['DOB'], errors='coerce')
data['Age'] = (datetime.now() - data['DOB']).dt.days // 365
data = data.drop(columns=['DOB'])

# 独热编码分类变量
data = pd.get_dummies(data, columns=['Gender', '10board', '12board', 'Degree', 'Specialization', 'CollegeState'], drop_first=True)

# 标准化数值特征
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_columns.remove('Salary')
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# 分割数据集
X = data.drop(columns=['Salary'])
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 进行简化的超参数搜索和模型训练
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("最佳参数:", best_params)
print("最佳R2分数:", best_score)

# 使用最佳参数训练模型
best_xgb = XGBRegressor(**best_params)
best_xgb.fit(X_train, y_train)

# 在测试集上评估模型
y_pred = best_xgb.predict(X_test)
test_r2_score = r2_score(y_test, y_pred)

print("测试集R2分数:", test_r2_score)

