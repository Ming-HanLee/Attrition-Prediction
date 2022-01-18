from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from resources.predict import Predictor

app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/predict/*": {"Access-Control-Allow-Origin": "http://localhost:3000"}})
        
##
## Actually setup the Api resource routing here
##
api.add_resource(Predictor, '/predict/<mode>')


if __name__ == '__main__':
    app.run(debug=True)