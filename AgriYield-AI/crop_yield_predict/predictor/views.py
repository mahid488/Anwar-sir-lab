from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import joblib
import pandas as pd
from .forms import CropYieldForm

def predict_yield(request):
    if request.method == 'POST':
        form = CropYieldForm(request.POST)
        if form.is_valid():
            nitrogen = form.cleaned_data['nitrogen']
            phosphorus = form.cleaned_data['phosphorus']
            potassium = form.cleaned_data['potassium']
            temperature = form.cleaned_data['temperature']
            humidity = form.cleaned_data['humidity']
            ph = form.cleaned_data['ph']
            rainfall = form.cleaned_data['rainfall']
            crop = form.cleaned_data['crop']

            model = joblib.load(settings.BASE_DIR / 'model' / 'crop_model.joblib')
            le = joblib.load(settings.BASE_DIR / 'model' / 'label_encoder.joblib')

            crop_encoded = le.transform([crop])[0]
            input_data = pd.DataFrame(
                [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, crop_encoded]],
                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
            )

            prediction = model.predict(input_data)[0]
            return JsonResponse({'predicted_yield': prediction})
        else:
            return JsonResponse({'error': form.errors.as_json()}, status=400)
    
    form = CropYieldForm()
    return render(request, 'index.html', {'form': form})