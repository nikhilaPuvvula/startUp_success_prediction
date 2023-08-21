from flask import Flask, render_template, request
import pandas as pd
import pickle
import warnings
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    warnings.filterwarnings("ignore")
    data = pd.read_csv("data.csv")
    data = data.loc[:,
           ['relationships', 'funding_rounds', 'funding_total_usd', 'milestones', 'avg_participants', 'status']]
    x = data.drop(['status'], axis=1)
    y = data['status']

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    x = sc.fit_transform(x)

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(bootstrap=False, max_depth=12, min_samples_leaf=100, min_samples_split=20,
                                n_estimators=100)
    rf.fit(x, y)

    pickle.dump(rf, open('prediction.pkl', 'wb'))
    prediction = pickle.load(open("prediction.pkl", 'rb'))
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        relationships = float(request.form['relationships'])
        funding_rounds =float( request.form['funding_rounds'])
        funding_total_usd = float( request.form['funding_total_usd'])
        milestones = float( request.form['milestones'])
        avg_participants = float( request.form['avg_participants'])
        data = [[float(relationships), float(funding_rounds), float(funding_total_usd), float(milestones),
                 int(avg_participants)]]
        ir = pickle.load(open('prediction.pkl', 'rb'))
        pred = ir.predict(data)[0]
        if relationships <= 0 and funding_rounds <= 0 and funding_total_usd <= 0 and milestones <= 0 and avg_participants <= 0:
            prediction = 'PLEASE GIVE VALID DATA'
        elif funding_total_usd <= 350000:
            prediction = "Our Prediction says that your startup might fail....But Don't Worry...Try to increase your relationships,Total funding,milestones to become successful...GOOD LUCK"
        elif pred=='closed':
            prediction = "Our Prediction says that your startup might fail....But Don't Worry...Try to increase your relationships,Total funding,milestones to become successful...GOOD LUCK"
        else:
            prediction='Our Prediction says that your startup will hopefully be a success....ALL THE BEST'
        return render_template('index.html', prediction=prediction, relationships=relationships,
                               funding_rounds=funding_rounds, funding_total_usd=funding_total_usd,
                               milestones=milestones, avg_participants=avg_participants)


if __name__ == '__main__':
    app.run(debug=True)
