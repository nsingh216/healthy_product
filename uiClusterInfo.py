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
    combined = pd.concat([test[["product_name", "Cluster_Number"]], train[["product_name", "Cluster_Number"]]])
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


def getClusterTag(clusterNum):
    """
    what is label is associated with this cluster?

    :param clusterNum: cluster label
    :return:
    """
    healthyClusterTags = {1: "LOW SAT FAT", 2: "LOW CHOLESTEROL"}

    # get the keys from the cluster tags dictionary
    keys = healthyClusterTags.keys()

    if clusterNum in healthyClusterTags:
        tag = healthyClusterTags[clusterNum]
    else:
        tag = ""

    return tag