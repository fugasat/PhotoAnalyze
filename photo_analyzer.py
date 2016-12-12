import pandas as pd
import numpy as np


def read_csv():
    df = pd.read_csv("data/instagram_data.csv")

    # nanを0に変換
    df.ix[:, "c":"踏切"] = df.ix[:, "c":"踏切"].fillna(0)

    # 登録された車両形式を抽出
    type_df = df.iloc[:, 17:23]
    type_array = type_df.as_matrix().flatten()
    type_array = type_array.astype('str')
    type_array = np.unique(type_array)
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

    # 風景の特徴が無いデータのカラムを追加
    df["特徴なし"] = no_feature

    return df

if __name__ == "__main__":
    data = read_csv()
    print(data)
    pass
