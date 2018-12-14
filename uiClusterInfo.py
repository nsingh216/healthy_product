import pandas as pd


def loadClusters():
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    combined = pd.concat([test[["product_name", "Cluster_Number"]], train[["product_name", "Cluster_Number"]]])
    return combined

def getCluster(cluster_df, product):

    cluster = cluster_df.loc[cluster_df["product_name"] == product]
    return cluster["Cluster_Number"]


def getClusterTag(clusterNum):
    healthyClusterTags = {1: "LOW SAT FAT", 2: "LOW CHOLESTEROL", 7: "A TAG" , 15: "ANOTHER", 17: "ONE", 19: "HI"}

    #keys = healthyClusterTags.keys()
    if clusterNum in healthyClusterTags:
        tag = healthyClusterTags[clusterNum]
    else:
        tag = ""

    return tag