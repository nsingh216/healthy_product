import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


##########################
#### Idea from: https://www.cambridge.org/core/journals/public-health-nutrition/article/comparison-of-heuristic-and-modelbased-clustering-methods-for-dietary-pattern-analysis/355233434504C5EC749927B49D95E2F7/core-reader
# Greve, B., Pigeot, I., Huybrechts, I., Pala, V., & Börnhorst, C. (2016).
# A comparison of heuristic and model-based clustering methods for dietary pattern analysis.
# Public Health Nutrition, 19(2), 255-264. doi:10.1017/S1368980014003243


############################################################################
## Data sourced from: https://www.kaggle.com/openfoodfacts/world-food-facts
## Created a function for my preprocessing steps so I could easily rerun as list of columns was updated
##
## This function takes in the large provided tsv file and returns a file with data relevant to my tool
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
"""

"""

def setListofProducts():

    product_name = pd.read_csv("./data/proc_products_us.csv")

    product_name['product_name'].replace(' ', np.nan, inplace=True)
    product_name = product_name[["product_name"]].dropna()

    product_name["product_name"].drop_duplicates(inplace=True)

    product_name = product_name["product_name"].sort_values(ascending = True)
    product_name.to_csv("./data/sorted_products_us.csv", header = True, index=False)

    return product_name


def getResultsForUser(searchTerm="apple"):
    df_all_products = pd.read_csv("./data/sorted_products_us.csv")
    products = calculateCosineSimilarity_withTD_IDF(df_all_products, searchTerm= searchTerm)
    return products


"""
def getSimilarProducts():
    product_name = pd.read_csv("./data/sorted_products_us.csv")

    product_name_list = product_name['product_name'].tolist()[0:99]
    product_name_list2 = product_name['product_name'].tolist()[0:99]
    #print(product_name_list)

    column_names = ['product1', 'product2', 'similarity_full', 'similarity_nouns']
    similarity_df = pd.DataFrame(columns=column_names)

    for i in product_name_list:
        tokens_prod1 = nltk.word_tokenize(i.lower())
        pos_prod1 = nltk.pos_tag(tokens_prod1)
        print(pos_prod1)

        noun_tags = ['NN', 'NNS', 'PDT']
        verb_tags = ['VBD']
        other_tags = ['CD']

        nouns_i = ""
        nouns_and_verbs = ""

        for token, pos in pos_prod1:
            if pos in noun_tags:
                nouns_i = nouns_i + token + " "


        for j in product_name_list2:
            nouns_j = ""
            tokens_prod2 = nltk.word_tokenize(j.lower())
            pos_prod2 = nltk.pos_tag(tokens_prod2)
            print(pos_prod2)

            for token, pos in pos_prod2:
                if pos in noun_tags:
                    nouns_j = nouns_j + token + " "


            scoreFull = fuzz.ratio(i, j)
            scoreNouns = fuzz.ratio(nouns_i, nouns_j)
            #print(i + " " + j + " " + str(score))
            aList = [i, j, scoreFull, scoreNouns]
            row = pd.Series(aList, index = column_names)
            similarity_df =similarity_df.append(row, ignore_index=True)
            print(similarity_df)

    similarity_df.to_csv("./data/similarNames2.csv", header=True)
    return 1
"""

def calculateCosineSimilarity_withTD_IDF(product_name, searchTerm = ""):

    #product_name = pd.read_csv("./data/sorted_products_us.csv")


    vec = TfidfVectorizer(analyzer="word")

    # returns a sparse matrix with tf-idf weighted doc terms
    tdidf_matrix = vec.fit_transform(product_name["product_name"])

    similarities = get_similar_products(tdidf_matrix, selected_item= -1, user_product=searchTerm)
    products = get_product_names(similarities, product_name)

    return products


def get_product_names(similarities, product_name):
    products = []

    for i in similarities:
        index = i[0]
        products.append(product_name["product_name"][index])

    return products

def get_similar_products(tfidf_matrix, selected_item = -1, user_product = "chocolate", top_n = 10):

    # compare existing products in dataframe or use a new product
    if selected_item == -1:
        match_string = user_product
    else:
        match_string = tfidf_matrix[selected_item:selected_item + 1]


    ## cosine similarity was running really slow and resulting in memory issues so using linear kernel
    ## linear kernel = cosine similarity when using td-idf normalized vectors
    ## returns 1D array
    similarity = linear_kernel(match_string, tfidf_matrix).flatten()

    ## reverse order of the indices would result in a sorted array of similar products
    indices_of_similar_products = similarity.argsort()[::-1]

    ## return list of similar product indices and the similarity score
    similar_products= []
    for index in indices_of_similar_products:
        if index != selected_item:
            similarity_score = similarity[index]
            similar_products.append((index, similarity_score))

    #indices_of_similar_products = [i for i in indices_of_sorted_ndarry[::-1] if i != selected_term]

    return similar_products[0:top_n]


############################################################################


#preprocessing()

#calculateCosineSimilarity_withTD_IDF()

#setListofProducts()
#getSimilarProducts()




