import pandas as pd


def read_csv():
    df = pd.read_csv("data/instagram_data.csv")

    # nanを0に変換
    df.ix[:, "c":"踏切"] = df.ix[:, "c":"踏切"].fillna(0)

    # 風景の特徴が無いデータを「特徴なし」とする
    no_feature = []
    for i, row in df.iterrows():
        no_feature.append(row["森林" : "踏切"].sum())

    df["特徴なし"] = no_feature

    return df

if __name__ == "__main__":
    data = read_csv()
    print(data)
    pass
