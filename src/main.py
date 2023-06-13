import os.path

# 导入可视化库
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 导入数据挖掘库
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

project_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(project_path, "data")
# pd 设置
pd.options.display.float_format = "{:.2f}".format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 1000)


def read_data():
    """
    读取数据
    :return:
    """
    # 读取数据并去重, 去掉 nan 值, 去掉未知性别的数据
    df = pd.read_csv(os.path.join(data_path, "diabetes_prediction_dataset.csv"))
    duplicate_rows = df[df.duplicated()].shape[0]
    print("重复行数:", duplicate_rows)
    df = df.drop_duplicates().dropna()
    df = df[df['gender'] != 'Other']
    print("各列的统计数据:")
    print(df.describe().style.format("{:.2f}").data)
    return df


def preparation(df):
    """
    预处理
    :param df:
    :return:
    """

    def map_smoking(smoking_status):
        """
        重新对smoking_history进行取值
        :param smoking_status:
        :return:
        """
        if smoking_status in ['never', 'No Info']:
            return 'non_smoker'
        elif smoking_status == 'current':
            return 'current_smoker'
        elif smoking_status in ['ever', 'former', 'not current']:
            return 'past_smoker'

    df['smoking_history'] = df['smoking_history'].apply(map_smoking)

    # 各属性分布情况
    # df['smoking_history'].value_counts().to_csv(os.path.join(data_path, "smoking_history_dist.csv"), index_label=False)
    # df['age'].value_counts().to_csv(os.path.join(data_path, "age_dist.csv"), index_label=False)
    # df['gender'].value_counts().to_csv(os.path.join(data_path, "gender_dist.csv"), index_label=False)
    # df['hypertension'].value_counts().to_csv(os.path.join(data_path, "hypertension_dist.csv"), index_label=False)
    # df['heart_disease'].value_counts().to_csv(os.path.join(data_path, "heart_disease_dist.csv"), index_label=False)

    # 年龄分布柱状图
    plt.hist(df['age'], bins=30, edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

    # 吸烟历史分布柱状图
    sns.countplot(x='smoking_history', data=df)
    plt.title('Smoking History Distribution')
    plt.show()

    # 性别分布饼图
    g_series = df['gender'].value_counts()
    plt.pie(g_series.tolist(), labels=g_series.index.tolist())
    plt.title('Gender Distribution')
    plt.show()

    # 高血压分布饼图
    g_series = df['hypertension'].value_counts()
    plt.pie(g_series.tolist(), labels=g_series.index.tolist())
    plt.title('Hypertension Distribution')
    plt.show()

    # 心脏病分布饼图
    g_series = df['heart_disease'].value_counts()
    plt.pie(g_series.tolist(), labels=g_series.index.tolist())
    plt.title('Heart disease Distribution')
    plt.show()

    print("label字段分布情况:")
    diabetes_df = df["diabetes"].value_counts()
    diabetes_df.to_csv(os.path.join(data_path, "diabetes_dist.csv"), index_label=False)
    print(diabetes_df)

    # label 字段分布不均匀, 重新取样
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)

    # 对数值字段进行 z-score 标准化, 对离散字段进行 one-hot 编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(),
             ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']),
            ('cat', OneHotEncoder(), ['gender', 'smoking_history'])
        ])

    # 创建数据处理pipeline, 使用随机森林分类算法
    clf = imbPipeline(steps=[('preprocessor', preprocessor),
                             ('over', over),
                             ('under', under),
                             ('classifier', RandomForestClassifier())])

    # 定义多组超参数, 用于确定最优的超参
    # classifier__n_estimators: random forest 中决策树的数量
    # classifier__max_depth: 决策树最大深度
    # classifier__min_samples_split: 非叶子节点最小数据量
    # classifier__min_samples_leaf: 叶子节点最小数据量
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # 创建 grid search 对象, 使用交叉验证
    grid_search = GridSearchCV(clf, param_grid, cv=5)

    return grid_search


def train_and_predict(df, grid_search):
    # 将数据分为测试集和训练集
    # df = df[:1000]
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 开始训练
    grid_search.fit(X_train, y_train)
    print("最优超参数: ", grid_search.best_params_)

    # 不同超参数效果折线图
    cv_df = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=cv_df, x='param_classifier__n_estimators', y='mean_test_score', hue='param_classifier__max_depth',
                 palette='viridis')
    plt.title('Hyperparameters Tuning Results')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean Test Score')
    plt.show()

    # 预测和评估
    y_pred = grid_search.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=["TN", "FP"], index=["FN", "TP"])

    # 混淆矩阵图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


if __name__ == '__main__':
    df = read_data()
    grid_search = preparation(df)
    train_and_predict(df, grid_search)
