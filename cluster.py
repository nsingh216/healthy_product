import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

"""
Determining the optimum number of clusters using silhouette analysis (SA)

SA determines the distance between different clusters. If two or more clusters are very close to each other, 
then there is a greater chance the border points might have been assigned to the wrong cluster.

Reference:
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
"""
def getNumClusters(df, attributes):

    scores = []
    for num_clusters in range(2, 25):

        GM = GaussianMixture(
            n_components=num_clusters
        )

        # using 80:20 train test ratio -- 70-80% = general recommendation for train data.
        training_data, test_data = train_test_split(df, test_size=0.2, random_state=0)

        ## remove the product name -- not something we are using for the clustering
        X_train = training_data[attributes].values

        #print(np.isnan(X_train).any())
        #print(np.isinf(X_train).any())

        # run model
        model = GM.fit(X_train)

        out = model.predict(X_train)
        # Mean= " + str(model.means_)

        print(str(num_clusters) + " clusters;" + " #iter= " + str(model.n_iter_) + "converged: " + str(model.converged_))


        silhouette_avg = silhouette_score(X_train, out)
        print("For n_clusters =", num_clusters, "The average silhouette_score is :", silhouette_avg)

        tup = (num_clusters, silhouette_avg, GM.bic(X_train))
        scores.append(tup)
        print(scores)

        # sample_silhouette_values = silhouette_samples(X_train, out)
        # print(str(sample_silhouette_values))





def normalize_data(ingredient_attribute, prediction_attributes):
    """
    My clusters were not differenciating well, and after some reading, I realized that this was because the data is not a
    normal/gaussian distribution. GMM expects the input data to follow a normal distribution

    Testing distribution of the features:

    The plots illustrustrate the non-normal distribution of each column. They appear to be more geometric or loglike,
    so we need to normalize them. Reference: http://www.kmdatascience.com/2017/07/box-cox-transformations-in-python.html

    After some more reading and testing, I learned about the box-cox transformation method and decided to use it to transform
    the columns.
    Commenting out the plot in the for loop for submission because sometimes it leads to memory issues with so many plots.

    Sci-py has some built -in box cox transformations, so I decided to use that.

    """
    for i in ingredient_attribute[prediction_attributes].columns:

        """plot = sns.distplot(ingredient_attribute[i], hist=True, kde=False,
                 bins=int((ingredient_attribute[i].max() - ingredient_attribute[i].min())/10), color='green',
                 hist_kws={'edgecolor':'black'})
        # clear the plot otherwise each iteration of the loop will place a new graph on top
        plt.clf()
        """

        # clean the data  --
        # from Open Food Facts: https://static.openfoodfacts.org/data/data-fields.txt
        # "fields that end with _100g correspond to the amount of a nutriment (in g, or kJ for energy) for 100 g or 100 ml of product"
        if i.endswith("_100g"):
            # based on the field description, the range of these columns = 0 to 100
            # drop columns with neg values
            ingredient_attribute = ingredient_attribute.drop(ingredient_attribute[ingredient_attribute[i] < 0].index)
            # and those > 100
            ingredient_attribute = ingredient_attribute.drop(ingredient_attribute[ingredient_attribute[i] > 100].index)


        # get values to transform
        transform = np.asarray(ingredient_attribute[i].values)
        # boxcox requires strictly positive values (> 0), so resetting zeros to a small pos #
        getZeros = transform[transform < 1] = 1
        transformed_data = stats.boxcox(transform)[0]

        # save back the transformed data
        ingredient_attribute[i] = transformed_data

        # how many bins should the histogram plot have? calculated using range of each column
        bins = int((transformed_data.max() - transformed_data.min()) / 10)
        if bins < 3:
            bins = 5

        """
        ## replot to see the difference
        plot = sns.distplot(transformed_data, hist=True, kde=False,
                 bins= bins, color='orange',
                 hist_kws={'edgecolor':'black'})

        # and clear again
        plt.clf()
        """
    ingredient_attribute.to_csv("./data/transformed_data_us.csv")
    return ingredient_attribute


def getNumZeros(df, cols):
    """
    Calculate the % of zeros:

    Pre box cox transformation:
        energy_100g: 4.87124550268828
        fat_100g: 33.786866406019904
        saturated-fat_100g: 33.770696296467406
        cholesterol_100g: 51.25116222662408
        carbohydrates_100g: 10.695949965061013
        sugars_100g: 18.0781824796863
        fiber_100g: 30.425793634751876
        proteins_100g: 27.86398627850704
        salt_100g: 15.433792063941233
        sodium_100g: 15.433792063941233
        calcium_100g: 29.359721412112567
        iron_100g: 24.557776378934967

    Post:
        energy_100g: 21.657633242999097
        fat_100g: 96.61246612466125
        saturated-fat_100g: 99.66124661246613
        cholesterol_100g: 100.0
        carbohydrates_100g: 31.07497741644083
        sugars_100g: 55.19421860885276
        fiber_100g: 68.92502258355917
        proteins_100g: 77.64227642276423
        salt_100g: 78.77145438121048
        sodium_100g: 89.25022583559169
        calcium_100g: 99.77416440831075
        iron_100g: 100.0

    Some of the columns have a large number of zeros, so that might be a reason that the clusters are not working

    """

    df_rows = len(df)

    for col_name in df[cols].columns:
        numZeros = len(df[df[col_name] == 0])
        zero_percent = 100 * float(numZeros)/float(df_rows)
        print(col_name + ": " + str(zero_percent))


def process_file():
    ingredient_info = pd.read_csv("./data/proc_products_us.csv")

    #############################################################################
    # Cluster the different products based on the nutrition information
    ############################################################################

    # some of the nutrient values are sparse -- only want to drop rows where there is missing values for columns we will use
    attributes = ["product_name",
                  "energy_100g",
                  "fat_100g",
                  "saturated-fat_100g",
                  "trans-fat_100g",
                  "cholesterol_100g",
                  "carbohydrates_100g",
                  "sugars_100g",
                  "fiber_100g",
                  "proteins_100g",
                  "salt_100g",
                  "sodium_100g",
                  "calcium_100g",
                  "iron_100g",
                  "nutrition-score-fr_100g",
                  "nutrition-score-uk_100g"
                  ]

    # the clusters are still not looking great, need to try some different features

    ingredient_attribute = ingredient_info[attributes]

    # models in sklearn tend to have issues with missing values -- drop for now
    ingredient_attribute = ingredient_attribute.dropna()

    # apply same cleaning effort as in product names section
    ingredient_attribute["product_name"].replace(' ', np.nan, inplace=True)
    ingredient_attribute["product_name"] = ingredient_attribute[["product_name"]].dropna()
    ingredient_attribute["product_name"].drop_duplicates(inplace=True)
    return ingredient_attribute



# ##################### EXECUTION STARTS HERE #########################
if __name__ == "__main__":

    # commented because we don't need to process the file each time
    df = process_file()

    # dont want to use product name as a predictor -- should really be using the list above with indices
    # but this is easier to switch around as I test different attributes
    prediction_attributes = ["energy_100g",
                             "fat_100g",
                             "saturated-fat_100g",
                             "cholesterol_100g",
                             "carbohydrates_100g",
                             "sugars_100g",
                             "fiber_100g",
                             "proteins_100g",
                             "salt_100g",
                             "sodium_100g",
                             "calcium_100g",
                             "iron_100g"
                             ]

    # commented because we don't need to normalize the data each time
    # normalized_df = normalize_data(df, prediction_attributes)

    # read file instead
    # normalized_df = pd.read_csv("./data/transformed_data_us.csv")
    # getNumZeros(normalized_df, prediction_attributes)

    # update prediction attributes based on percent of zeros
    prediction_attributes = ["energy_100g",
                             "carbohydrates_100g",
                             "sugars_100g",
                             "fiber_100g",
                             "proteins_100g",
                             "salt_100g"
                             ]

    getNumClusters(df, prediction_attributes)

    ########################################################################################################################
    """
    My understanding of the Guassian Mixture Model:

    """
    ########################################################################################################################

    num_clusters = 4

    GM = GaussianMixture(
            n_components=num_clusters
    )

    # using 80:20 train test ratio -- 70-80% = general recommendation for train data.
    training_data, test_data = train_test_split(df, test_size=0.2, random_state=0)

    # remove the product name -- not something we are using for the clustering
    X_train = training_data[prediction_attributes].values
    X_test = test_data[prediction_attributes].values

    # run model

    # print(np.isnan(X_train).any())
    # print(np.isinf(X_train).any())

    # X_train= pd.DataFrame(X_train, columns=prediction_attributes).dropna()
    model = GM.fit(X_train)

    out = model.predict(X_train)
    # Mean= " + str(model.means_)
    print(str(num_clusters) + " clusters;" + " #iter= " + str(model.n_iter_) +
          "converged: " + str(model.converged_))


    # store back predictions of clusters after making
    training_data["Cluster_Number"] = model.predict(X_train)
    test_data["Cluster_Number"] = model.predict(X_test)


    # save off the data to csv files
    training_data.to_csv("./data/train.csv")
    test_data.to_csv("./data/test.csv")






