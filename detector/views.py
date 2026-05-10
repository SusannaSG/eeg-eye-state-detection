from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.conf import settings
import joblib
import json
import os
import numpy as np

# Load models and metadata once when server starts
MODELS_DIR = settings.MODELS_DIR

def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree':       'decision_tree.pkl',
        'Random Forest':       'random_forest.pkl',
        'SVM':                 'svm.pkl',
        'KNN':                 'knn.pkl',
        'Naive Bayes':         'naive_bayes.pkl',
    }
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

def load_scaler():
    path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_metadata():
    path = os.path.join(MODELS_DIR, 'metadata.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

# Register view
def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('predict')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

# Home view
def home(request):
    return redirect('predict')

# Predict view
@login_required
def predict(request):
    metadata = load_metadata()
    features = metadata['feature_names'] if metadata else []
    
    if request.method == 'POST':
        try:
            # Get input values from form
            input_data = [float(request.POST.get(f, 0)) for f in features]
            input_array = np.array(input_data).reshape(1, -1)

            # Scale input
            scaler = load_scaler()
            if scaler:
                input_scaled = scaler.transform(input_array)
            else:
                input_scaled = input_array

            # Predict with all models
            models = load_models()
            results = []
            best_model = metadata.get('best_model', '')

            for name, model in models.items():
                pred = model.predict(input_scaled)[0]
                label = 'Eyes Open' if pred == 1 else 'Eyes Closed'
                acc = metadata['results'][name]['accuracy']
                results.append({
                    'model': name,
                    'prediction': label,
                    'accuracy': acc,
                    'is_best': name == best_model,
                })

            # Sort by accuracy
            results.sort(key=lambda x: x['accuracy'], reverse=True)

            return render(request, 'result.html', {
                'results': results,
                'input_data': dict(zip(features, input_data)),
                'best_model': best_model,
            })

        except Exception as e:
            return render(request, 'predict.html', {
                'features': features,
                'error': str(e)
            })

    return render(request, 'predict.html', {'features': features})

# Result view
@login_required
def result(request):
    return redirect('predict')