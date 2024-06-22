import numpy as np 
import pandas as pd 
import os

# 遍历目录并打印文件路径
for dirname, _, filenames in os.walk('D:/20224329/基于XGBoost的印度工科毕业生的薪资预测/dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np

# 读取数据集
df = pd.read_csv('D:/20224329/基于XGBoost的印度工科毕业生的薪资预测/dataset/Engineering_graduate_salary.csv')

# 删除不必要的列
df.drop(['ID', 'DOB', 'CollegeID', '10board', '12graduation', '12board', 'CollegeState',
         'CollegeCityID', 'CollegeCityTier', 'GraduationYear'], axis=1, inplace=True)

# 删除重复项
df = df.drop_duplicates()

# 处理Specialization列
specialization = df.Specialization.value_counts(ascending=False)
specializationlessthan10 = specialization[specialization <= 10]

def removespeciallessthan10(value):
    if value in specializationlessthan10:
        return 'other'
    else:
        return value
df.Specialization = df.Specialization.apply(removespeciallessthan10)

# 处理collegeGPA列
df = df[(df['collegeGPA'] > 40)]

# 用平均值填补缺失值
df = df.replace(-1, np.nan)
cols_with_nan = [column for column in df.columns if df.isna().sum()[column] > 0]
for column in cols_with_nan:
    df[column] = df[column].fillna(df[column].mean())

# 对分类变量进行标签编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df.Gender = le.fit_transform(df.Gender)
df.Degree = le.fit_transform(df.Degree)
df.Specialization = le.fit_transform(df.Specialization)

# 切分数据集并进行标准化处理
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

x = df.drop('Salary', axis=1)
y = df['Salary']

sc = StandardScaler()
x = sc.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

# 训练XGBoost模型
xgb = XGBRegressor()
xgb.fit(X_train, y_train)

# 预测并计算R2分数
predictions = xgb.predict(X_test)
xgb_r2_score = xgb.score(X_test, y_test)

print("XGBoost R2 Score: ", xgb_r2_score)
