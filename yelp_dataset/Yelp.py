import json
import spacy
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

nlp = spacy.load("en_core_web_sm")

def preprocess_text_spacy(text):
    doc = nlp(text.lower())  # Lowercasing
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]  # Lemmatization & stopword removal
    return " ".join(tokens)

def find_bussiness_catergory(id):
    with open("yelp_academic_dataset_business.json", "r") as file:
        for line in file:
            review = json.loads(line)
            if id == review["business_id"]:
                return review["categories"]

# Read JSON file
with open("yelp_academic_dataset_review.json", "r") as file:
    for line in file:
        review = json.loads(line)  # Convert JSON string to a Python dictionary
        
        # Extract relevant fields
        review_id = review["review_id"]
        user_id = review["user_id"]
        business_id = review["business_id"]
        stars = review["stars"]
        text = review["text"]
        date = review["date"]

        
        catergory = find_bussiness_catergory(business_id)
        print(catergory)
        cleaned_text = [preprocess_text_spacy(text)]  
        print(cleaned_text)
        tagged_data = [TaggedDocument(words=text.split(), tags=[str(text)])]

        model = Doc2Vec(tagged_data, vector_size=50, window=2, min_count=1, workers=4, epochs=20)

        # new_review = "Loved the food, but service was bad" (data scraped review)
        # vector = model.infer_vector(preprocess_text_spacy(new_review).split()) (This vector can be used to compare )