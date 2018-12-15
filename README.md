# healthy_product
 Nivedita Singh
 nsingh10@illinois.edu
<br />
<br />

## Text Information Systems Project Overview
The idea behind this project is to cluster different food products based on the associated nutritional components. I used a dataset from Kaggle provided by the Open Food Facts Organization. This dataset contains a variety of attributes about over 600k food products from over 150 countries, such as names, brands, manufacturer, nutritional information, product source, availibity, etc. For the purposes of my project, I decided to focus my analysis on products available in the United States and on the nutritional components (preprocessing function in healthyIngredients.py)

Once I had extracted the data that I wanted to use, I thought about how I wanted to categorize the products. Since this dataset did not have labels, I needed to choose an unsupervised learning technique, such as clustering. I decided to use Gaussian Mixture Model (GMM) from the scikit-learn library in Python after reading the Comparision of Clustering Methods paper cited below. This paper noted that GMM enables clusters of varying volumes, unlike other popular techniques such as K-Means. To identify the number of clusters that I should use, I used the silhouette score, also available in scikit-learn.

I also created a simple UI component using the Flask Microframework in Python. The webpage is quite simple, with a search box where users can search for a product. Behind the scenes, I identify similarly named products using cosine similarity. In healthyIngredients.py, I decided to apply tf-idf weighting first to emphasize important words and de-emphasize common terms. To calculate the cosine similarity, I used linear_kernel also from scikit-learn, since using the available cosine similarity function was leading to performance issues. Since the product names are primarly short phrases, I noticed that adding in stopwords did not add much benefit. Once the top 10 similarily named products (to the original search term(s)) are identified, I obtain the cluster that each belongs to. The products that belong in the healhier clusters are returned to the user, again using flask.

<br />
<br />

## Installation/ Setup
 I have added all the currently installed packages from my projects to the requirements.txt file. (Some of the them were used it previous iterations of the solution, but I think it is the easiest way to install the packages.)
 ```
 pip install -r requirements.txt
 ```


<br />
<br />



 ## Running the code
 
 I used PyCharm to write, debug, and execute my code. There are two ways to run the code:
 
1. Run ```website.py```. Running this file will load the simple Flask website. For simplicity, I have been using localhost for my project. Copy the link () to a browser (Chrome). This will load the webpage with a search box - enter a product that you want to search for and click search. The next page will display a few healthy options. 

2. Run the function without a UI. The function that runs after the user clicks on the Search button is the ```getResultsForUser``` function in ```healthyIngredients.py```. This function takes one parameter - the search term. To run the non-website version of the code, uncomment line 330 in ```healthyIngredients.py``` and update the parameter with your search term. The provided example ```getResultsForUser("apple")``` searches for apple. It will print out the products to the console. 
 
 Video URL: 
 
<br />
<br />


## Resources:
  **Data**: https://www.kaggle.com/openfoodfacts/world-food-facts <br />
  **Process/ Idea**: https://www.cambridge.org/core/journals/public-health-nutrition/article/comparison-of-heuristic-and-modelbased-clustering-methods-for-dietary-pattern-analysis/355233434504C5EC749927B49D95E2F7 <br />
          Greve, B., Pigeot, I., Huybrechts, I., Pala, V., & Börnhorst, C. (2016). A comparison of heuristic and model-based clustering methods for dietary pattern analysis. Public Health Nutrition, 19(2), 255-264. doi:10.1017/S1368980014003243
