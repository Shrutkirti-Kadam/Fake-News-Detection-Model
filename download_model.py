from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Specify the model name
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Download and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("./distilbert_sentiment")

# Download and save model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained("./distilbert_sentiment")

print("Model and tokenizer downloaded and saved to './distilbert_sentiment'")
