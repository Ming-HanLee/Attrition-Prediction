from typing import Dict
from flask_restful import Resource, reqparse
from model.model import Model
import pandas as pd

class Predictor(Resource):
    model = Model()
    parser = reqparse.RequestParser()
    parser.add_argument('data', required=True, type=dict, action='append', help='Please provide at least one employee information')

    def post(self, mode):
        arg = self.parser.parse_args()
        # print(type(arg['data']))
        # print(arg['data'])
        response = {'message': 'Success'}

        x = pd.DataFrame(arg['data'])

        if mode == 'plot':
            probs, img_base64 = self.model.predict(x, mode)
            response['result'] = {
                'probs': probs,
                'img': img_base64.decode()
            }
        else:
            probs = self.model.predict(x, mode)
            response['result'] = {
                'probs': probs
            }

        # predictions = self.__filter(probs)

        return response, 200

    def __filter(self, probs, thres=0.5):
        
        # 手動產生預測結果
        predicted_y = []
        for p in probs:
            if p >= thres:
                predicted_y.append(1)
            else:
                predicted_y.append(0)
        
        return predicted_y
