import pandas as pd


def read_csv():
    df = pd.read_csv("data/instagram_data.csv")
    df.ix[:, "c":"踏切"] = df.ix[:, "c":"踏切"].fillna(0)
    return df

if __name__ == "__main__":
    data = read_csv()
    print(data)
    pass
