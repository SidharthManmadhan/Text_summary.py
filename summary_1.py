from fastapi import FastAPI
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
# Load model 
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
app = FastAPI()
@app.get("/summaries")
def summary(Article:str):
    tokens = tokenizer(Article, truncation=True, padding="longest", return_tensors="pt")
    summ = model.generate(**tokens)
    summ = tokenizer.decode(summary)
    return summ
