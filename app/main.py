from flask import Flask, render_template, request, redirect
import requests
from bs4 import BeautifulSoup
from lxml import html
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
import warnings
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
warnings.filterwarnings("ignore")
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PredictionDetails(db.Model):
    __tablename__ = "Prediction"
    article_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    url = db.Column(db.String(500))
    content = db.Column(db.Text)
    category = db.Column(db.String(50))



model: Pipeline = joblib.load("../model/category_predictor.sav")
Encoder = joblib.load("../model/Encoder.sav")


def predict(text):
    y_pred = model.predict(pd.Series(text))[0]
    y_pred = Encoder.inverse_transform(y_pred.reshape((-1,1)))[0]
    print(y_pred)
    return y_pred


def filtering(text):
    def check_nltk_resource(resource):
        try:
            nltk.data.find(resource)
        except LookupError:
            return False

        return True


# Initialize NLTK
    nltk_resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for nltk_res in nltk_resources:
        if not check_nltk_resource(nltk_res):
            nltk.download(nltk_res)

    stop_words = set(stopwords.words('english'))

    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(text)
    filtered_text = [w for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_text)



def scraper(url):
    response = requests.get(url=url)
    # soup = BeautifulSoup(html,'lxml')
    if response.status_code == 200:
        tree = html.fromstring(response.content)
        elements = tree.xpath("//p//text()")
        text = " ".join(elements)
        if text != '':
            return predict(filtering(text.strip())),text.strip()


    raise Exception
# dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
# load_dotenv(dotenv_path)

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///database.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '#1010100'
db.init_app(app)
app.app_context().push()
db.create_all()

@app.route("/", methods=["GET"])
def getdata():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def get_prediction():
    print(request)
    url = request.form.get("url")
    try:
        result, content = scraper(url)
        new_prediction = PredictionDetails(url=url,category=result,content=content)
        db.session.add(new_prediction)
        print("added")
        db.session.commit()
        return (
            render_template(
                "result.html",
                url=url,
                result = result,
                content=content,
                id=new_prediction.article_id
            ),
            201,
        )

    except:
        return render_template("ServerError.html"), 500


@app.route("/results")
def show_result():
    data = PredictionDetails.query.all()
    return render_template("allresults.html", data=data)


@app.route("/delete/<string:id>", methods=["GET"])
def delete_result(id: str):
    data = PredictionDetails.query.get(id)  # Retrieve the record with the specified id
    if data:
        db.session.delete(data)  # Delete the record from the database
        db.session.commit()  # Commit the changes
    return redirect("/", code=302)


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("ServerError.html"), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template("error.html"), 404


if __name__ == '__main__':
    app.run(debug=True,port=5000)