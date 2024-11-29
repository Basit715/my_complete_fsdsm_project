from src.DiamondPricePrediction.pipelines.prediction_pipeline import customData,PredictPipeline
from flask import Flask,request,render_template,jsonify


app = Flask(__name__)

@app.route('/')
def home_page():
     return render_template('index.html')



@app.route('/preict', methods=["GET", "POST"])
def predict_datapoints():
     if request.method == "GET":
          render_template('form.html')
          
     else:
          data = customData(carat=float(request.form.get("carat")),
            depth=float(request.form.get("depth")),
            table=float(request.form.get("table")),
            x=float(request.form.get("x")),
            y=float(request.form.get("y")),
            z=float(request.form.get("z")),
            cut=request.form.get("cut"),
            color=request.form.get("color"),
            clarity=request.form.get("clarity"))
          
          
          final_data=data.get_data_as_dataframe()
          
          predict_pipeline=PredictPipeline()
          
          pred=predict_pipeline.predict(final_data)
          
          result=round(pred[0],2)
          
          return render_template("result.html",final_result=result)


app.run(debug=True)