# Fake News Detection Model (Worked With [Ashay Gupta](https://github.com/ashay1311) , [Rehan George Varghese](https://github.com/Rehangv) , [Vaibhav Hingnekar](https://github.com/vaibhavh27))
Made 2 Models for the detection of Fake News.  
The model uses CNN + biLSTM for news detection.  
Uses distilBERT for Sentiment Analysis.
## Download Links for 1st Model (Contains Databases, Model, and Tokenizer)
***[CLICK ME](https://drive.google.com/drive/folders/189UjfsBH5Ur6fOx6Q4VChhIJ3O4lQ-bf?usp=sharing)***
## Download Links for 2nd Model (Contains Databases, Model, and Tokenizer)
***[CLICK ME](https://drive.google.com/drive/folders/1Czjijig-OMXdrBfXNBii2d13Wfs3KmUh?usp=sharing)***

## Building Your Model
### Download Requirements
```
pip install pandas
pip install numpy
pip install nltk
pip install spacy
pip install imbalanced-learn
pip install scikit-learn
pip install tensorflow
pip install transformers
python -m spacy download en_core_web_sm
```
```
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')
```
### Files In Repo and What They Keep
1. Code - Consists of Code for model building. Can also use your database for creating your model.
2. Download your model - Consists of code to download your model and tokenizer.
3. Quick Load Model - Code for quick running of the model ( load your model and tokenizer and run in a separate file )

<!-- # Fake News Detection Model
Made 2 Models for the detection of Fake News.  
The model uses CNN + biLSTM for news detection.  
Uses distilBERT for Sentiment Analysis.
## Download Links for 1st Model (Contains Databases, Model, and Tokenizer)
**[CLICK ME](https://drive.google.com/drive/folders/189UjfsBH5Ur6fOx6Q4VChhIJ3O4lQ-bf?usp=sharing)**
## Download Links for 2nd Model (Contains Databases, Model, and Tokenizer)
**[CLICK ME](https://drive.google.com/drive/folders/1Czjijig-OMXdrBfXNBii2d13Wfs3KmUh?usp=sharing)**

## Building Your Model
### Download Requirements

pip install pandas
pip install numpy
pip install nltk
pip install spacy
pip install imbalanced-learn
pip install scikit-learn
pip install tensorflow
pip install transformers
python -m spacy download en_core_web_sm


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

### Files In Repo and What They Keep
1. Code - Consists of Code for model building. Can also use your database for creating your model.
2. Download your model - Consists of code to download your model and tokenizer.
3. Quick Load Model - Code for quick running of the model ( load your model and tokenizer and run in a separate file ) -->



