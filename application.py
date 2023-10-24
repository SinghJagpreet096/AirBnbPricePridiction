from flask import Flask,render_template,request
import pandas as pd
from src.exception import CustomException
import sys
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

application = Flask(__name__)


@application.route('/',methods=['GET','POST'])
def index():
    df = pd.read_csv('notebook/train.csv')
    cities = list(df.smartlocation.unique())
    cancellationpolicy = list(df.cancellationpolicy.unique())
    roomtype = list(df.roomtype.unique())
    try:
        print("get")
        if request.method == 'GET':
            return render_template('index.html',cities=cities,
                            cancellationpolicy=cancellationpolicy,
                            roomtype=roomtype)
        else:
            print('post')
            data = CustomData(smartlocation=request.form.get('city'),
                            roomtype=request.form.get('roomtype'),
                            minimumnights=request.form.get('minimumnights'),
                            availability365=request.form.get('availability365'),
                            numberofreviews=request.form.get('numberofreviews'),
                            reviewscoresrating=request.form.get('reviewscoresrating'),
                            cancellationpolicy=request.form.get('cancellationpolicy'))
            
            pred_df = data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            results  = predict_pipeline.predict(pred_df)
            print(results)
            return render_template('index.html',results=results[0])
    except Exception as e:
        raise CustomException(e, sys)
    

# @application.route('/predict',methods=['GET','POST'])
# def predict_price():
#     try:
#         if request.method=='GET':
#             return render_template()
#     except Exception as e:
#         raise CustomException(e, sys)



if __name__ == "__main__":
    application.run(host='0.0.0.0',debug=True)

