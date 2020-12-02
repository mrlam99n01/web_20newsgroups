from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from rest_framework.parsers import JSONParser
from .models import Article
from .serializers import ArticleSerializer
from .serializers import ArticleSerializer
from django.views.decorators.csrf import csrf_exempt
from .naive_bayes import NaiveBayes
import json
import pandas as pd
import os
import requests
from .prediction_file import Prediction
from .predictionv2 import naive_bayes_probabilities
colnames=['catergories', 'labels']
# Create your views here.
prediction = Prediction()
api_url = 'https://tronglam1245.herokuapp.com/api/'
@csrf_exempt
def article_list(request):
    if request.method =='GET':
        articles = Article.objects.all()
        serialzer = ArticleSerializer(articles,many=True)
        #print(serialzer.data)
        return JsonResponse(serialzer.data,safe=False)

    elif request.method =='POST':
        data = JSONParser().parse(request)
        arr = []
        arr.append(data['field_name'])

        accuracy_dict = prediction.naive_bayes_probabilities(arr,data['model_name'])
        if isinstance(accuracy_dict,dict):
            for i in range(0,4):
                data['top_{}_accuracy'.format(i+1)] = accuracy_dict['top_{}_accuracy'.format(i+1)]
                data['top_{}_percentage'.format(i + 1)] = accuracy_dict['top_{}_percentage'.format(i + 1)]
            print("Naive")
        else:
            data['top_1_accuracy'] = accuracy_dict[0]
        serialzer = ArticleSerializer(data=data)

        if serialzer.is_valid():
            serialzer.save()
            return JsonResponse(serialzer.data,safe=False,status=201)
        return JsonResponse(serialzer.errors,status=400)

def index(request):
    response = requests.get(api_url).json()
    print(response)
    context = {
        'serialzerdata': response
    }
    print(context['serialzerdata'])
    return render(request, 'nlp_prj/nlp.html', context)
@csrf_exempt
def article_detail(request,pk):
    try:
        print("pk",pk)
        article = Article.objects.get(pk=pk)
        print(article)
    except Article.DoesNotExist:
        return HttpResponse(status=404)
    if request.method == 'GET':
        serializer = ArticleSerializer(article)
        print(serializer.data)
        return JsonResponse(serializer.data)
    elif request.method == 'DELETE':
        article.delete()
        return HttpResponse(status=204)
def test(request):
    return  render(request,'nlp_prj/copy_nlp.html')
def map_labels_featues(key=[]):
    value = key[0]
    colnames = ['catergories', 'labels']
    goal_dir = os.path.join(os.getcwd(),"nlp_prj","train.map")
    data = pd.read_csv(goal_dir, names=colnames, delimiter=" ")
    print("data :",data)
    print("value: ",value)
    print("goal dir",goal_dir)
    print("data.loc[data['labels'] ==value].values.tolist()",data.loc[data['labels'] ==value].values.tolist())
    return data.loc[data['labels'] ==value].values.tolist()[0][0]
if __name__=='__main__':
    print("day la file mainnnnnnnnnnnnnnn")