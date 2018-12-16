import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

import uiClusterInfo as cl



"""
Idea from: https://www.cambridge.org/core/journals/public-health-nutrition/article/comparison-of-heuristic-and-modelbased-clustering-methods-for-dietary-pattern-analysis/355233434504C5EC749927B49D95E2F7/core-reader
    Greve, B., Pigeot, I., Huybrechts, I., Pala, V., & Börnhorst, C. (2016).
    A comparison of heuristic and model-based clustering methods for dietary pattern analysis.
    Public Health Nutrition, 19(2), 255-264. doi:10.1017/S1368980014003243
"""

############################################################################
# Data sourced from: https://www.kaggle.com/openfoodfacts/world-food-facts
# Created a function for my preprocessing steps so I could easily rerun as list of columns was updated
#
# This function takes in the large provided tsv file and returns a file with data relevant to my tool
# (filter to products sold in the United States, remove attributes not related to nutrition)
############################################################################

def preprocessing():
    df = pd.read_csv("./data/products.tsv", sep='\t', low_memory=False)

    ## use only products sold in the us -- running into language issues with nltk pos tagging
    df.countries_en = df.countries_en.str.lower()
    # print(df.countries_en.unique() )

    df = df.loc[df.countries_en == 'united states']

    # REMOVE COLUMNS that are irrelevant to nutrition labels
    ingredient_info = df[["product_name", "generic_name", "quantity", "packaging", "packaging_tags", "brands",
                          "brands_tags",
                          "ingredients_text",
                          "serving_size",
                          "nutrition_grade_uk",
                          "nutrition_grade_fr",
                          "energy_100g",
                          "energy-from-fat_100g",
                          "fat_100g",
                          "saturated-fat_100g",
                          "monounsaturated-fat_100g",
                          "polyunsaturated-fat_100g",
                          "omega-3-fat_100g",
                          "omega-6-fat_100g",
                          "omega-9-fat_100g",
                          "trans-fat_100g",
                          "cholesterol_100g",
                          "carbohydrates_100g",
                          "sugars_100g",
                          "-sucrose_100g",
                          "-glucose_100g",
                          "-fructose_100g",
                          "-lactose_100g",
                          "-maltose_100g",
                          "-maltodextrins_100g",
                          "starch_100g",
                          "polyols_100g",
                          "fiber_100g",
                          "proteins_100g",
                          "casein_100g",
                          "serum-proteins_100g",
                          "nucleotides_100g",
                          "salt_100g",
                          "sodium_100g",
                          "alcohol_100g",
                          "vitamin-a_100g",
                          "beta-carotene_100g",
                          "vitamin-d_100g",
                          "vitamin-e_100g",
                          "vitamin-k_100g",
                          "vitamin-c_100g",
                          "vitamin-b1_100g",
                          "vitamin-b2_100g",
                          "vitamin-pp_100g",
                          "vitamin-b6_100g",
                          "vitamin-b9_100g",
                          "folates_100g",
                          "vitamin-b12_100g",
                          "biotin_100g",
                          "pantothenic-acid_100g",
                          "silica_100g",
                          "bicarbonate_100g",
                          "potassium_100g",
                          "chloride_100g",
                          "calcium_100g",
                          "phosphorus_100g",
                          "iron_100g",
                          "magnesium_100g",
                          "zinc_100g",
                          "copper_100g",
                          "manganese_100g",
                          "fluoride_100g",
                          "selenium_100g",
                          "chromium_100g",
                          "molybdenum_100g",
                          "iodine_100g",
                          "caffeine_100g",
                          "taurine_100g",
                          "ph_100g",
                          "fruits-vegetables-nuts_100g",
                          "fruits-vegetables-nuts-estimate_100g",
                          "collagen-meat-protein-ratio_100g",
                          "cocoa_100g",
                          "chlorophyl_100g",
                          "nutrition-score-fr_100g",
                          "nutrition-score-uk_100g",
                          "glycemic-index_100g"
                          ]]

    # REMOVE COLUMNS with high percents of NA values (> 80)
    nans = ingredient_info.isnull().sum() / ingredient_info.shape[0]
    #print(nans)


    ## these are the ones we will keep
    ingredient_info = df[[
        "product_name",
        "quantity",
        "serving_size",
        "nutrition_grade_fr",
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
        "vitamin-a_100g",
        "vitamin-c_100g",
        "calcium_100g",
        "iron_100g",
        "nutrition-score-fr_100g",
        "nutrition-score-uk_100g"
    ]]

    ### CONFIRM No columns with high percents of NA values remain (> 80)
    nans = ingredient_info.isnull().sum() / ingredient_info.shape[0]

    #### now we are left with products and their nutrient information
    #print(ingredient_info.head(10))

    ## nltk categorizes capitalized words a plural noun
    ingredient_info.product_name = ingredient_info.product_name.str.lower()

    """
    ## translate product names to english for nltk pos tagging
    for index, row in ingredient_info.iterrows():
        translator = Translator()
        eng_text = translator.translate(row["product_name"], dest="en").text
        row["product_name_en"] = eng_text
    """

    ## write to file so we can use this processed data
    ingredient_info.to_csv("./data/proc_products_us.csv")


############################################################################

def setListofProducts(product_name):
    """
    clean the list of product names after the preprocessing step. The names are then stored in a csv that is used later
    to identify products with similar names.

    :return: product_name: a dataframe with the cleaned up list of sorted product names
    """

    # read in the csv file with all the products and the nutrition attributes
    # product_name = pd.read_csv("./data/proc_products_us.csv")

    # for this step we only care about the product names
    product_name['product_name'].replace(' ', np.nan, inplace=True)

    # remove nans and duplicates, and sort the names
    product_name = product_name[["product_name"]].dropna()
    product_name["product_name"].drop_duplicates(inplace=True)
    product_name = product_name["product_name"].sort_values(ascending = True)

    #write out to csv
    product_name.to_csv("./data/sorted_products_us.csv", header = True, index=False)

    return product_name


def getResultsForUser(searchTerm):
    """
    Identify similar product names using TF-IDF weighting and cosine similarity as the similarity metric.

    :param searchTerm: string
    :return: df: a pandas dataframe with the products that have similar names when compared to the search term.
    """
    # load the results
    clusterData = cl.loadClusters()
    # only need to reload product after when new train/test.csv
    # df_all_products = pd.DataFrame(clusterData["product_name"])
    # setListofProducts(df_all_products)

    df_all_products = pd.read_csv("./data/sorted_products_us.csv")

    # use apple in case an empty string is passed
    if searchTerm.strip() == "":
        searchTerm = "apple"

    # add the search term to the top of the dataframe containing all the existing products
    data = []
    data.insert(0, {"product_name": searchTerm})
    df_all_products = pd.concat([pd.DataFrame(data), df_all_products], ignore_index=True)
    selected_index = 0

    # identify similarly named products using tf_idf weights and cosine similarity
    products = calculateCosineSimilarity_withTF_IDF(product_name=df_all_products, selected_index= selected_index)

    # Set up the dataframe to be returned to the user
    cols = ["Product", "Cluster_Number"]
    df = pd.DataFrame(columns=cols)

    # get the ranking of the average health score of the clusters
    means = cl.getClusterMeans(pd.DataFrame(clusterData[["Cluster_Number", "nutrition-score-uk_100g"]]))

    # which cluster does each of the similarly named product belong to?
    for i in products:
        clusterNum = cl.getCluster(clusterData, i)
        if np.isnan(clusterNum):
            clusterNum = 20
        df = df.append({"Product": i, "Cluster_Number": clusterNum}, ignore_index=True)

    # sort the top 10 products based on which cluster they belong to
    # join cluster avg scores to the product information
    df = df.set_index("Cluster_Number").join(means.set_index("Cluster_Number"), how="left",  lsuffix="_product")
    df.rename(columns={'nutrition-score-uk_100g': 'avg_nutrition-score-uk_100g'}, inplace=True)

    # join in the other attributes
    df = df.set_index("Product").join(clusterData.set_index("product_name"), how="left", lsuffix="_product")

    # sort healthiest at top
    df = df.sort_values(by="avg_nutrition-score-uk_100g", ascending=False)
    df = df.reset_index()
    #df = df["Product"]

    print(df)
    return df


def calculateCosineSimilarity_withTF_IDF(product_name, selected_index = -1):
    """
    get the 10 most similar product names using cosine similarity and tf-idf weighting

    :param product_name: pandas dataframe with one column for the product names
    :param selected_index: the index of the term we are trying to find similarities for
    :return: products: a dataframe with 10 similarily named products
    """

    # get tf_idf weights
    tfidf_matrix = get_TF_IDF_matrix(product_name)

    #
    similarities = get_similar_products(tfidf_matrix= tfidf_matrix, selected_item= selected_index)
    products = get_product_names(similarities, product_name)

    return products


def get_TF_IDF_matrix(product_name):
    """
    Use the TF-IDF Vectorizer in scikit-learn to obtain the tf-idf weights of each product

    :param product_name: pandas dataframe with all the product names
    :return: tfidf_matrix: a sparse matrix with tf-idf weighted doc terms
    """

    """ use the tf_idf vectorizer in sklearn
        - want features to be full words instead of strings
        - smoothing is true by default
    """
    vec = TfidfVectorizer(analyzer="word")

    # returns a sparse matrix with tf-idf weighted doc terms
    tfidf_matrix = vec.fit_transform(product_name["product_name"])

    return tfidf_matrix


def get_similar_products(tfidf_matrix, selected_item = 1, top_n = 10):
    """
    Compute similarity between each of the product names

    :param tfidf_matrix: sparse matrix with tf-idf weights
    :param selected_item: what is the index of the search term; default = 1
    :param top_n: number of similar product names to be returned
    :return: similar_products: list contains tuples of the indices of similar products and the similarity score
    """

    # compare existing products in dataframe or use a new product
    sel_vector =tfidf_matrix[0:1]

    """
    cosine similarity was running really slow and resulting in memory issues so using linear kernel
        linear kernel = cosine similarity when using tf-idf normalized vectors
        returns 1D array
    """
    dot_poduct = linear_kernel(sel_vector, tfidf_matrix).flatten()

    # reverse order of the indices would result in a sorted array of similar products
    indices_of_similar_products = dot_poduct.argsort()[::-1]

    # return list of similar product indices and the similarity score
    similar_products= []
    for index in indices_of_similar_products:
        if index != selected_item:
            similarity_score = dot_poduct[index]
            similar_products.append((index, similarity_score))


    return similar_products[0:top_n]


def get_product_names(similarities, product_name):
    """

    :param similarities: list contains tuples of the indices of similar products and the similarity score
    :param product_name: pandas dataframe containing the actual names of the products
    :return: products: list of the similar product names
    """
    products = []

    for i in similarities:
        index = i[0]
        products.append(product_name["product_name"][index])

    return products


############################################################################

###### EXECUTION STARTS HERE #########
if __name__ == "__main__":

    # processing on the original file to get to a file that will work for this project
    # commented because this doesnt need to run each time since the data is saved in a csv.
    # preprocessing()


    # this is the function that is called by the flask webpage. The parameter is the search term that the user searched for.

    getResultsForUser("apple")