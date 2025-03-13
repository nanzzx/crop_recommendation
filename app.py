from flask import Flask, request, render_template
import numpy as np
from predictor import predict_fertility_changes_and_recommend_crop

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None  
    error = None    

    if request.method == 'POST':
        try:
            # Get Before Cultivation Inputs
            n_before = float(request.form['n_before'])
            p_before = float(request.form['p_before'])
            k_before = float(request.form['k_before'])
            ph_before = float(request.form['ph_before'])
            temp = float(request.form['temp'])
            humidity = float(request.form['humidity'])
            rainfall = float(request.form['rainfall'])

            # Get After Cultivation Inputs (Allow Empty)
            cultivated_crop = request.form.get('cultivated_crop', "").strip()
            n_after = request.form.get('n_after', "").strip()
            p_after = request.form.get('p_after', "").strip()
            k_after = request.form.get('k_after', "").strip()
            ph_after = request.form.get('ph_after', "").strip()

            # If After Cultivation inputs are missing, assume no crop was planted
            if not cultivated_crop or not n_after or not p_after or not k_after or not ph_after:
                cultivated_crop = None
                n_after, p_after, k_after, ph_after = n_before, p_before, k_before, ph_before

            else:
                n_after = float(n_after)
                p_after = float(p_after)
                k_after = float(k_after)
                ph_after = float(ph_after)

            # Call Prediction Function
            results = predict_fertility_changes_and_recommend_crop(
                n_before, p_before, k_before, ph_before,
                n_after, p_after, k_after, ph_after,
                cultivated_crop, temp, humidity, rainfall)

        except ValueError as ve:
            error = f"Invalid input: {ve}"
        except Exception as e:
            error = f"Unexpected error: {e}"

    return render_template('index.html', results=results, error=error)

if __name__ == '__main__':
    app.run(debug=True)
