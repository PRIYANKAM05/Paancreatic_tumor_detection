from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
import pickle
import numpy as np
import re

from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from nltk.stem import WordNetLemmatizer
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from django.views.decorators.csrf import csrf_exempt
import json
import random
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.models import load_model
from django.conf import settings
import os
loaded_model = load_model('Model\\Model.h5')

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

# Load TF-IDF model, corpus, and dataframe
try:
    tfidf_fit = joblib.load('Model\\tfidf_vectorizer.pkl')
    tfidf_corpus = joblib.load('Model\\tfidf_corpus.pkl')
    df = pd.read_csv('Model\\questions_answers.csv')
except FileNotFoundError as e:
    print(f"File loading error: {e}")




# Utility function to clean and preprocess text
def clean_data(text):
    text = re.sub(r"[\([{})\]]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def HomePage(request):
    return render(request, 'home.html')

def SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        if pass1 != pass2:
            return HttpResponse("Your password and confirm password are not same!!")
        else:
            my_user = User.objects.create_user(uname, email, pass1)
            my_user.save()
            return redirect('login')

    return render(request, 'signup.html')

def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        pass1 = request.POST.get('pass')
        user = authenticate(request, username=username, password=pass1)
        if user is not None:
            login(request, user)
            return redirect('index1')
        else:
            return HttpResponse("Username or Password is incorrect!!!")

    return render(request, 'login.html')

def LogoutPage(request):
    logout(request)
    return redirect('home')

@login_required(login_url='login')
def index1(request):
    return render(request, 'index1.html')

@login_required(login_url='login')
def index(request):
    return render(request, 'index.html')

# def result(request):
#     return render(request, 'result.html')
def getPredictions(patient_cohort, sample_origin, age, sex, plasma_CA19_9, creatinine, LYVE1, REG1B,  TFF1, REG1A):
    model = pickle.load(open('Model\\rf.pkl', 'rb'))
    prediction = model.predict(np.array([[patient_cohort, sample_origin, age, sex, plasma_CA19_9, creatinine, LYVE1, REG1B,  TFF1, REG1A]]))
    return (prediction[0])

def result(request):
    patient_cohort = int(request.GET['patient_cohort'])
    sample_origin = int(request.GET[ 'sample_origin'])
    age = int(request.GET[ 'age'])
    sex = int(request.GET[ 'sex'])
    plasma_CA19_9 = float(request.GET[ 'plasma_CA19_9'])
    creatinine = float(request.GET[ 'creatinine'])
    LYVE1 = float(request.GET[ 'LYVE1'])
    REG1B = float(request.GET[ 'REG1B'])
    TFF1 = float(request.GET[   'TFF1'])
    REG1A = float(request.GET[ 'REG1A'])
   
    result = getPredictions(patient_cohort, sample_origin, age, sex, plasma_CA19_9, creatinine, LYVE1, REG1B,  TFF1, REG1A)
    if result.round()==1:
        res='Mild Tumore'
    elif result.round()==2:
        res='Severe Tumore'
    else:
        res='Normal'
    return render(request, 'result.html', {'result': res})

def predictImage(request):
    if request.method == 'POST' and request.FILES.get('filePath'):
            # Save the uploaded file
            fileObj = request.FILES['filePath']
            fs = FileSystemStorage()
            filePathName = fs.save(fileObj.name, fileObj)
            filePath = os.path.join(settings.MEDIA_ROOT, filePathName)

            # Preprocess the image for the model
            try:
                image = load_img(filePath, target_size=(224, 224))
                image = np.expand_dims(image, axis=0)

                # Predict the class
                prediction = loaded_model.predict(image)
                predicted_class = np.argmax(prediction)
                class_names = ['Normal', 'Pancreatic_Tumor']
                
                predict=class_names[predicted_class]
                context = {'filePathName': fs.url(filePathName), 'predictedLabel': predict}
                return render(request, 'index1.html', context)
            except Exception as e:
                return HttpResponse(f"An error occurred during prediction: {str(e)}")
    return HttpResponse("Invalid request")
@login_required(login_url='login')
@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        try:
            # Decode the JSON data from the request body
            data = json.loads(request.body.decode('utf-8'))
            print("Received data:", data)  # Log incoming data

            # Check if the 'user_input' field is present
            user_input = data.get('user_input', '').strip()
            if not user_input:
                return JsonResponse({'error': 'Missing or empty "user_input" field'}, status=400)

            # Define greeting responses
            GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
            GREETING_RESPONSES = ["hi there", "hello", "Hi, I am glad! You are talking to me"]

            # Check for greetings
            if any(word in user_input.lower() for word in GREETING_INPUTS):
                response_text = random.choice(GREETING_RESPONSES).capitalize()
            elif user_input in ['thanks', 'thank you']:
                response_text = "You are welcome."
            elif user_input == 'what is your name?':
                response_text = "I am a chatbot."
            elif user_input == 'bye':
                response_text = "Bye! Take care."
            else:
                # Clean and process the user input
                user_input_cleaned = clean_data(user_input)

                # Transform the user input into TF-IDF features
                tfidf_test = tfidf_fit.transform([user_input_cleaned])
                cosine_similarities = cosine_similarity(tfidf_test, tfidf_corpus).flatten()

                # Find the highest similarity index
                highest_similarity_index = cosine_similarities.argmax()

                # Check if there is a meaningful match
                if cosine_similarities[highest_similarity_index] == 0:
                    response_text = "I'm sorry, I don't have an answer for that."
                else:
                    # Get the answer from the dataframe
                    response_text = df.iloc[highest_similarity_index]['Answer'].capitalize()

        except Exception as e:
            # Handle any unexpected errors
            print(f"Error processing request: {e}")
            return JsonResponse({'error': 'An unexpected error occurred'}, status=500)

        # Return the chatbot response as JSON
        return JsonResponse({'response': response_text})

    # Return BadRequest if the method is not POST
    return HttpResponseBadRequest("Only POST method is allowed.")

@login_required(login_url='login')
def index2(request):
    return render(request, 'chatbot.html')