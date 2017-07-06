import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import KFold


def read_csv():
    df = pd.read_csv("data/instagram_data.csv")

    # 不要な列を削除
    #df = df.drop("低い", axis=1)
    #df = df.drop("近い", axis=1)

    # nanを0に変換
    df.ix[:, "f":"JR"] = df.ix[:, "f":"JR"].fillna(0)

    # 日付文字列をdatetimeに変換
    df["日付"] = pd.to_datetime(df["日付"])

    # 登録された車両形式を抽出
    type_df = df.iloc[:, 27:33].fillna("0") # 17〜23列を取得
    type_array = type_df.as_matrix().flatten() # 1次配列に変換
    type_array = type_array.astype('str') # 文字列に変換
    type_array = np.unique(type_array) # 重複値を除外
    type_array = np.sort(type_array) # ソート
    type_array = np.delete(type_array, 0)
    type_dic = {}
    for item in type_array:
        type_dic[item] = []

    no_feature = []
    no_area = []
    main_model = []
    for i, row in df.iterrows():
        # 風景の特徴が無いデータを「特徴なし」とする
        feature_count = row["森林":"踏切"].sum()
        if feature_count > 0:
            feature_count = 0
        else:
            feature_count = 1
        no_feature.append(feature_count)

        # 地域の特徴が無いデータを「特徴なし」とする
        area_count = row["北海道":"九州"].sum()
        if area_count > 0:
            area_count = 0
        else:
            area_count = 1
        no_area.append(area_count)

        # 特定の車両が存在するかどうかチェック
        types = row[27:33].astype('str')
        first_model = None
        for item in type_array:
            exists = item in types.as_matrix()
            if first_model is None and exists:
                first_model = item
            array = type_dic[item]
            array.append(int(exists))
            type_dic[item] = array

        # 先頭の車両データをメインの車両とする
        main_model.append(first_model)

    # 車両名を格納する列を削除
    df = df.iloc[:, :27]

    # 特徴が無いデータのカラムを追加
    df["scene特徴なし"] = no_feature
    df["area特徴なし"] = no_area
    df["main_model"] = main_model

    # 特定の車両が存在するかどうか表すカラムを追加
    for key, value in type_dic.items():
        df[key] = value

    return df


def linalg(data):
    x = data["日付"]
    x = np.array([[v.timestamp(), 1] for v in x])
    y = data["f"]
    m,c = np.linalg.lstsq(x, y)[0]
    return m, c


def good_data(data, y_lr_e):
    y_lr_e_std = y_lr_e.std()
    y_mask = y_lr_e >= y_lr_e_std
    return data.ix[y_mask]


def create_dataset(data, y_lr_e, num_train=70):
    y_lr_e_std = y_lr_e.std() / 2  # いいね数が標準偏差を超えたものを「人気データ」とラベリングする
    #y_lr_e_std = y_lr_e.mean()  # いいね数が平均を超えたものを「人気データ」とラベリングする
    y_mask = y_lr_e >= y_lr_e_std

    data = data.drop("日付", axis=1)
    data = data.drop("f", axis=1)
    data = data.drop("c", axis=1)
    data = data.drop("main_model", axis=1)

    pos_data = data.ix[y_mask]
    neg_data = data.ix[~y_mask]

    pos_label = np.ones(pos_data.shape[0])
    neg_label = np.zeros(neg_data.shape[0])

    pos_label_e = y_lr_e[y_mask]
    neg_label_e = y_lr_e[~y_mask]

    pos_index = np.arange(len(pos_data))
    random.shuffle(pos_index)
    pos_train_index = pos_index[:num_train]
    pos_test_index = pos_index[num_train:]

    neg_index = np.arange(len(neg_data))
    random.shuffle(neg_index)
    neg_train_index = neg_index[:num_train]
    neg_test_index = neg_index[num_train:]

    x = data.as_matrix()
    y = np.array([1 if y >= y_lr_e_std else 0 for y in y_lr_e.as_matrix()])
    y_e = y_lr_e.as_matrix()
    train_x = pd.concat([pos_data.iloc[pos_train_index, :], neg_data.iloc[neg_train_index, :]]).as_matrix()
    train_y = np.hstack((pos_label[pos_train_index], neg_label[neg_train_index]))
    train_y_e = pd.concat([pos_label_e.iloc[pos_train_index], neg_label_e.iloc[neg_train_index]]).as_matrix()
    test_x = pd.concat([pos_data.iloc[pos_test_index, :], neg_data.iloc[neg_test_index, :]]).as_matrix()
    test_y = np.hstack((pos_label[pos_test_index], neg_label[neg_test_index]))
    test_y_e = pd.concat([pos_label_e.iloc[pos_test_index], neg_label_e.iloc[neg_test_index]]).as_matrix()

    print("pos:" + str(len(pos_data)))
    print(" -train:" + str(len(pos_train_index)))
    print(" -test:" + str(len(pos_test_index)))
    print("neg:" + str(len(neg_data)))
    print(" -train:" + str(len(neg_train_index)))
    print(" -test:" + str(len(neg_test_index)))
    return x, y, y_e, train_x, train_y, train_y_e, test_x, test_y, test_y_e


def bad_data(data, y_lr_e):
    y_lr_e_std = y_lr_e.std()
    y_mask = y_lr_e >= -y_lr_e_std
    return data.ix[y_mask]


def train_pls(train_x, train_y, test_x, test_y):
    print()
    print("**** Train : PLS ****")

    # 学習
    clf = PLSRegression(n_components=15, scale=True)
    clf.fit(train_x, train_y_e)

    # 学習結果を確認
    test_pred = clf.predict(test_x)
    print("test_pred" + str(test_pred))
    print_coef(clf)


def train_cv(clf, x, y):
    """
    Cross Validation
    :param clf:
    :param x:
    :param y:
    :return:
    """
    print("- Train CV -")
    scores = []
    cv = KFold(len(x), n_folds=10)

    for train, test in cv:
        x_train, y_train = x[train], y[train]
        x_test, y_test = x[test], y[test]
        clf.fit(x_train, y_train)
        scores.append(clf.score(x_test, y_test))
        pass

    print("Mean(scores)=%.5f Stddev(scores)=%.5f" % (np.mean(scores), np.std(scores)))
    return clf


def train(clf, train_x, train_y, test_x, test_y):
    print("- Train -")
    # 学習
    clf.fit(train_x, train_y)

    # 学習結果を確認
    test_pred = clf.predict(test_x)
    print(classification_report(test_y, test_pred))
    print("accuracy_score = " + str(accuracy_score(test_y, test_pred)))
    print()


def print_coef(clf):
    coef = clf.coef_
    data_coef = data.drop("ID", axis=1)
    data_coef = data_coef.drop("日付", axis=1)
    data_coef = data_coef.drop("f", axis=1)
    data_coef = data_coef.drop("c", axis=1)
    col = data_coef.columns
    index = 0
    for c in col:
        if coef[0][index] > 0.4:
            print("(!)" + c + ":" + str(coef[0][index]))
        else:
            print("   " + c + ":" + str(coef[0][index]))
        index = index + 1

    print()
    print("Ranking")
    score_dict = {}
    for i, row in data.iterrows():
        id = row["ID"]
        score = 0
        index = 0
        for c in col:
            feature = row[c]
            score += feature * coef[0][index]
            index = index + 1
            pass
        score_dict[id] = score

    score_rank = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)
    for rank in range(10):
        print(score_rank[rank])


def train_svm(x, y, train_x, train_y, test_x, test_y):
    print()
    print("**** Train : SVM ****")
    clf = svm.SVC()
    train(clf, train_x, train_y, test_x, test_y)
    train_cv(clf, x, y)


def train_random_forest(x, y, train_x, train_y, test_x, test_y):
    print()
    print("**** Train : Random Forest ****")
    clf = RandomForestClassifier(n_estimators=100)
    train(clf, train_x, train_y, test_x, test_y)
    train_cv(clf, x, y)


def train_kn(x, y, train_x, train_y, test_x, test_y):
    print()
    print("**** Train : KNeighbors ****")
    clf = KNeighborsClassifier(n_neighbors=2)
    train(clf, train_x, train_y, test_x, test_y)
    train_cv(clf, x, y)


def train_logistic_regression(x, y, train_x, train_y, test_x, test_y):
    print()
    print("**** Train : LogisticRegression ****")
    clf = LogisticRegression()
    train(clf, train_x, train_y, test_x, test_y)
    train_cv(clf, x, y)
    print_coef(clf)


if __name__ == "__main__":
    data = read_csv()

    #
    # いいねの数を正規化する
    #

    # (x軸)日付を整数に変換
    x = data["日付"]
    x = np.array([v.timestamp() for v in x])

    # (y軸)いいねの数を取得
    y = data["f"]

    # 回帰直線を取得
    m, c = linalg(data)
    y_lr = m * x + c

    # 回帰直線といいね数の差分を取得する
    # (回帰直線が水平になるように(y軸)いいね数を調整する)
    y_lr_e = y - (m * x + c)

    # 差分の平均と分散を取得する
    y_lr_e_mean = y_lr_e.mean()  # ※正規化後の平均なので0に近い値になる
    y_lr_e_std = y_lr_e.std()

    plt.plot(x, y, "o")
    plt.plot(x, y_lr)
    plt.savefig('date_f.png')
    plt.clf()

    plt.plot(x, x * 0 + y_lr_e_mean, color="#808080")
    plt.plot(x, x * 0 + y_lr_e_std, color="#ff4040")
    plt.plot(x, x * 0 - y_lr_e_std, color="#ff4040")
    plt.plot(x, y_lr_e)
    plt.savefig('date_e.png')

    print()

    # 教師データ、テストデータ生成
    x, y, y_e, train_x, train_y, train_y_e, test_x, test_y, test_y_e = create_dataset(data, y_lr_e)

    # 学習
    #train_kn(x, y, train_x, train_y, test_x, test_y)
    train_svm(x, y, train_x, train_y, test_x, test_y)
    #train_random_forest(x, y, train_x, train_y, test_x, test_y)
    #train_logistic_regression(x, y, train_x, train_y, test_x, test_y)

    print()

    # 回帰誤差を２番目の列に挿入する
    data["回帰誤差"] = y_lr_e
    columns = data.columns.values
    columns = np.delete(columns, np.where(columns == "回帰誤差"))
    columns = np.insert(columns, 1, "回帰誤差")
    data = data.ix[:, columns]
    all = data.sort_values(by="ID", ascending=True)
    all.to_csv("./data/instagram_data_all.csv", index=False)

    good = good_data(data, y_lr_e)
    good = good.sort_values(by="回帰誤差", ascending=False)
    good.to_csv("./data/instagram_data_good.csv", index=False)

    bad = bad_data(data, y_lr_e)
    bad = bad.sort_values(by="回帰誤差", ascending=True)
    bad.to_csv("./data/instagram_data_bad.csv", index=False)

    pass