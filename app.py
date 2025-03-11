from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# with open('.qodo/tfidf_vectorizer.pkl', 'rb') as file:
#     vectorizer = pickle.load(file)

model = BertForSequenceClassification.from_pretrained("bert_model")
tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None 

    if request.method == 'POST':
        user_input = request.form['text']

        if user_input.strip(): # If input is not empty
            # Tokenize input text
            inputs = tokenizer(user_input, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()} 

            # Get model prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=1).item()

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
