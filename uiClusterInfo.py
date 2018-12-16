import pandas as pd


def loadClusters():
    """
    Combine the training and test datasets

    :return: combined: a pandas dataframe containing both the datasets
    """
    # load in files into dataframes
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    # combine the 2 dataframes

    attributes = ["product_name",
                  "energy_100g",
                  "carbohydrates_100g",
                  "sugars_100g",
                  "nutrition-score-fr_100g",
                  "nutrition-score-uk_100g",
                  "nutrition_grade_fr",
                  "Cluster_Number"
                  ]

    combined = pd.concat([test[attributes], train[attributes]])
    return combined


def getCluster(cluster_df, product):
    """
    :param cluster_df: the combined dataframe of the train and test datasets
    :param product: string; name of the product
    :return: cluster number (int)
    """

    # get the row that corresponds with this product
    cluster = cluster_df.loc[cluster_df["product_name"] == product]
    return cluster["Cluster_Number"].max()

def getClusterMeans(df):
    """

    :param df: dataframe with cluster label and uk nutrition scores
    :return: ranking of clusters
    """

    avgs = df.groupby(["Cluster_Number"], as_index=False).mean().sort_values(by="nutrition-score-uk_100g", ascending=True)
    return avgs