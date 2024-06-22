# 导入必要的库
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

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

# 初步训练XGBoost模型并进行特征选择
initial_xgb = XGBRegressor(random_state=42)
initial_xgb.fit(X_train, y_train)
selector = SelectFromModel(initial_xgb, threshold="mean", prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print("原始特征数:", X_train.shape[1])
print("选择后的特征数:", X_train_selected.shape[1])

# 定义扩展的参数网格进行超参数搜索
extended_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

extended_grid_search = GridSearchCV(estimator=initial_xgb, param_grid=extended_param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
extended_grid_search.fit(X_train_selected, y_train)

extended_best_params = extended_grid_search.best_params_
extended_best_score = extended_grid_search.best_score_

print("扩展后的最佳参数:", extended_best_params)
print("扩展后的最佳R2分数:", extended_best_score)

# 使用扩展后的最佳参数训练模型
extended_best_xgb = XGBRegressor(**extended_best_params)
extended_best_xgb.fit(X_train_selected, y_train)

# 在测试集上评估模型
extended_y_pred = extended_best_xgb.predict(X_test_selected)
extended_test_r2_score = r2_score(y_test, extended_y_pred)

print("扩展后的测试集R2分数:", extended_test_r2_score)

# 定义基础模型进行堆叠
base_models = [
    ('xgb1', XGBRegressor(**extended_best_params)),
    ('xgb2', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3))
]

# 定义堆叠模型
stacked_model = StackingRegressor(estimators=base_models, final_estimator=Ridge())

# 训练堆叠模型
stacked_model.fit(X_train_selected, y_train)

# 在测试集上评估堆叠模型
stacked_y_pred = stacked_model.predict(X_test_selected)
stacked_test_r2_score = r2_score(y_test, stacked_y_pred)

print("堆叠模型的测试集R2分数:", stacked_test_r2_score)
