#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
from P3_model_deployment import predict_proba


app = Flask(__name__)

api = Api(app,  version='1.0', title='Movie Classifier API',  description='Movie Classifier API')
ns = api.namespace('predict', description='Movie Classifier API')
   
parser = api.parser()


parser.add_argument('plot', type=str, required=True, help='Plot to be analyzed', location='args')


resource_fields = api.model('Resource', {    'result': fields.String,})
@ns.route('/')
class MovieApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        
        plot= args['plot']
       
        
        return {
         "result": predict_proba(plot)
        }, 200

        
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
