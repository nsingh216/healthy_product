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
 
1. Run ```website.py```. Running this file will load the simple Flask website. For simplicity, I have been using localhost for my project. Copy the link (127.0.0.1:5000) to a browser (Chrome). This will load the webpage with a search box - enter a product that you want to search for and click search. The next page will display a few healthy options. 

2. Run the function without a UI. The function that runs after the user clicks on the Search button is the ```getResultsForUser``` function in ```healthyIngredients.py```. This function takes one parameter - the search term. To run the non-website version of the code, uncomment line 330 in ```healthyIngredients.py``` and update the parameter with your search term. The provided example ```getResultsForUser("apple")``` searches for apple. It will print out the products to the console. 
 
 Video URL: 
 
<br />
<br />



##  **Files:**
  1. cluster.py: This file contains 4 functions that I used to cluster the data into 20 different clusters using GMM.
             <br /> - *getNumClusters*: calculates the BIC score and the silhouette score for a range of the number of clusters, to identify what number of clusters I should use in the final model run.
             <br /> - *normalize_data*: GMM requires a normal distribution for the input data. Since the features I was using did not follow a normal distribution, I applied a box cox transformation from Python's sci-py in an effort to improve the quality of the clusters.
             <br /> - *getNumZeros*: What percentage of each feature is zero? Some of the features had a large number of zeros that I felt were a factor in creating poor clusters. 
             <br /> - *process_file*: cleans the file that holds the nutrition facts for the US products
             <br /> - *main section*: split cleaned data into train and test datasets, applies GMM and saves the data into CSV files
  2. healthyIngredients.py: 
             <br /> - *preprocessing*: load products.tsv (file from Kaggle -- this file is too large for me to upload to github); Filters to US products and removes the unneeded features. Saves processed data to proc_products_us.csv
             <br /> - *setListofProducts*: cleans the product names (remove duplicates, NaNs, sorts and saves into sorted_products_us.csv)
             <br /> - *getResultsForUser*: this is the function that is called from the webpage. It identifies similarly named products, the cluster labels of the products and returns the dataframe with the data to the webpage. 
  3. uiClusterInfo.py:             
            <br /> - *getCluster*: returns the cluster that the product belongs to
            <br /> - *getClusterMeans*: returns the mean health score of the cluster
            <br /> - *loadClusters*: load the files that has the cluster labels from cluster.py
  4. website.py:
            <br /> - *home*: loads index.html (starting webpage)
            <br /> - *results*: loads search.html (webpage with search results)

  5. data (directory)
  6. templates (directory)-- these files contain the html and css files used by Flask for the web component of this solution
      <br /> - *index.html*: landing page for the UI
      <br /> - *search.html*: the search results are returned here
      <br /> - *styles.css*: style sheet for the website
  7. requirements.txt: contains list of python package for this project

<br /> 
<br /> 

## **Concepts:**
* **Gaussian Mixture Model:** 
GMM is similar to the K-Means clustering algorithm that we learned about in this course (select k mean points and assign the remaining points to the closest mean. Iteratively select new means by minimizing the sum of square distance measure.) GMM is a probabilistic model, and accounts for the existence of randomness by returning a probability distribution of the possible results rather than just one deterministic result. It also enable the creation of non-spherical clusters unlike K-Means. It also applies the expectation-maximization process that we learned about in this course (e-step = calculate the weights associated with the probability that the data point belongs to each of the available clusters; m-step = Obtain new parameters based on the weights in the E-step; repeat until the model converges to [local] minima.)

* **BIC Score: (Bayesian Information Criterion)** The BIC score applies the Bayesian principle and determines what is the posterior probability of the model being true? The lower the BIC score the greater the chance of it being true. The BIC score is categorized as a penalty likelihood score, meaning that it penalizes complicated models to account for overfitting.

* **Silhouette Score:** The silhouette score determines the quality of the clusters. It returns a value between -1 and 1 and higher scores are indicative of better clusters. To determine the quality of the clusters, it uses a distance measure (points close to the edges of multiple clusters have a greater chance of being mapped to the wrong cluster.)

* **TF IDF Weighting: (Term Frequency Inverse Document Frequency)** Term frequency is the count of the number of occurrence of a term. Long documents or particularly frequent words can give the wrong picture of the important words in a text. The IDF portion determines how frequent the term is across the entire corpus and gives greater weight to the less common words. 


* **Cosine Similarity:** Distance measure to determine how similar the text or product names were. Each product name can be visualized as a vector of words and each word is a different dimension. The similarity is determined by the angle between the two vectors. Two exactly same vectors will overlap and result in an angle = 0 between them (cos(0) = 1) and result in a similarity score of 1.

<br /> 
<br /> 


## Resources:
  **Data**: https://www.kaggle.com/openfoodfacts/world-food-facts <br />
  **Process/ Idea**: https://www.cambridge.org/core/journals/public-health-nutrition/article/comparison-of-heuristic-and-modelbased-clustering-methods-for-dietary-pattern-analysis/355233434504C5EC749927B49D95E2F7 <br />
          Greve, B., Pigeot, I., Huybrechts, I., Pala, V., & Börnhorst, C. (2016). A comparison of heuristic and model-based clustering methods for dietary pattern analysis. Public Health Nutrition, 19(2), 255-264. doi:10.1017/S1368980014003243
