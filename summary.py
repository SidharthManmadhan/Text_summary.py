from fastapi import FastAPI
from transformers import pipeline
summarizer = pipeline("summarization")
app = FastAPI()
@app.get("/summary")
def summary(Article:str):

 summ= summarizer(Article, max_length=130, min_length=30, do_sample=False)
 return summ
