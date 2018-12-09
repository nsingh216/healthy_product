from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import pandas as pd


ingredient_info = pd.read_csv("./data/proc_products_us.csv")

#############################################################################
## Cluster the different products based on the nutrition information
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
              "iron_100g"
              ]

# dont want to use product name as a predictor -- should really be using the list above with indices
# but this is easier to switch around as I test different attributes
prediction_attributes = ["energy_100g",
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
                         "iron_100g"
                         ]

ingredient_attribute = ingredient_info[attributes]

# models in sklearn tend to have issues with missing values -- drop for now
# todo: try imputing values and see how accuracy compares
ingredient_attribute = ingredient_attribute.dropna()


########################################################################################################################
"""
My understanding of the Guassian Mixture Model:

"""
########################################################################################################################
GM = GaussianMixture(
    n_components=19
)


# using 80:20 train test ratio for now
training_data, test_data = train_test_split(ingredient_attribute, test_size=0.2, random_state=0)

X_train = training_data[prediction_attributes].values
X_test = test_data[prediction_attributes].values

# run model
model = GM.fit(X_train)

# store back predictions of clusters after making
training_data["Cluster_Number"] = model.predict(X_train)
test_data["Cluster_Number"] = model.predict(X_test)


# save off the data
training_data.to_csv("./data/train.csv")
test_data.to_csv("./data/test.csv")