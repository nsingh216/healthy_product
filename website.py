from flask import Flask, render_template, request
import healthyIngredients as hi
import pandas


"""
References: 
 - https://medium.freecodecamp.org/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492
 

"""

app = Flask(__name__)
app._static_folder="./templates/"
default_term = "apple"

## Landing page ~ index.html
@app.route("/")
def home():
    #session["searchTerm"] = request.form.get('search', default_term)
    return render_template('index.html')


### go to results page
@app.route('/results', methods=['POST', 'GET'])
def results():
    #searchTerm = session.get("searchTerm", default_term)
    if request.method == 'POST':
        result = request.form
        results = hi.getResultsForUser(result['search'])

        html_data = pandas.DataFrame(results).to_html()
        return render_template('search.html', tables=[html_data], titles=['na', 'Health Scores'])


if __name__ == "__main__":
    #app.secret_key = config.flask_secret_key
    app.run(debug=True)
