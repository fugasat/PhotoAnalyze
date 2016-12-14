import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_csv():
    df = pd.read_csv("data/instagram_data.csv")

    # nanを0に変換
    df.ix[:, "c":"踏切"] = df.ix[:, "c":"踏切"].fillna(0)

    # 日付文字列をdatetimeに変換
    df["日付"] = pd.to_datetime(df["日付"])

    # 登録された車両形式を抽出
    #df.ix[:, 17:23] = df.ix[:, 2:23].fillna(0)
    type_df = df.iloc[:, 17:23].fillna("0") # 17〜23列を取得
    type_array = type_df.as_matrix().flatten() # 1次配列に変換
    type_array = type_array.astype('str') # 文字列に変換
    type_array = np.unique(type_array) # 重複値を除外
    type_array = np.sort(type_array) # ソート
    type_array = np.delete(type_array, 0)
    type_dic = {}
    for item in type_array:
        type_dic[item] = []

    no_feature = []
    for i, row in df.iterrows():
        # 風景の特徴が無いデータを「特徴なし」とする
        no_feature.append(row["森林" : "踏切"].sum())
        # 特定の車両が存在するかどうかチェック
        types = row[17:23].astype('str')
        for item in type_array:
            exists = item in types.as_matrix()
            array = type_dic[item]
            array.append(int(exists))
            type_dic[item] = array

    # 車両名を格納する列を削除
    df = df.iloc[:, :17]

    # 風景の特徴が無いデータのカラムを追加
    df["特徴なし"] = no_feature

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


if __name__ == "__main__":
    data = read_csv()
    m, c = linalg(data)

    x = data["日付"]
    x = np.array([v.timestamp() for v in x])
    y = data["f"]
    plt.plot(x, y, "o")
    plt.plot(x, (m * x + c))
    plt.savefig('date_f.png')
    pass
