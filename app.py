from flask import Flask, redirect, url_for, request, render_template
import pickle
import pandas as pd
import numpy as np
import hts
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import altair as alt
from io import BytesIO
import base64
warnings.simplefilter("ignore")

app = Flask(__name__)

# load the data
df = pd.read_csv("Natural Gas.csv")
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df['Date'] = df['Date'].dt.date
df.columns = [col_name.lower() for col_name in df.columns]
df_state_level = df.groupby(["date", "state"]).sum().reset_index(drop=False).pivot(index="date", columns="state", values="consumption")
df_total = df.groupby("date")["consumption"].sum().to_frame().rename(columns={"consumption": "total"})
# join the DataFrames
hierarchy_df = df_state_level.join(df_total)
hierarchy_df.index = pd.to_datetime(hierarchy_df.index)
hierarchy_df = hierarchy_df.resample("MS").sum()


def ValuePredictor(state, steps_ahead):
    loaded_model = pickle.load(open("auto-arima.pckl", "rb"))
    result = loaded_model.predict(steps_ahead=steps_ahead)
    return result

@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        state=request.form['state']
        steps_ahead=int(request.form['shortterm'])
        predictions_df = ValuePredictor(state, steps_ahead)
        table_df = pd.DataFrame()
        table_df = predictions_df[[state]].copy()
        table_df = table_df.tail(steps_ahead)
        fig, ax = plt.subplots()
        predictions_df[state].plot(ax=ax, label="Predicted")
        hierarchy_df[state].plot(ax=ax, label="Observed")
        ax.legend()
        ax.set_title(state)
        ax.set_xlabel("Year")
        ax.set_ylabel("Consumption");
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return render_template('index.html',  tables=[table_df.to_html(classes='data')], titles=table_df.columns.values, plot_url=plot_url)


if __name__ == '__main__':
    app.run(debug=True)
