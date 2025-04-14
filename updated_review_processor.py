import json
import os
import spacy
import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import re
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("review_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ReviewProcessor")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class YelpDataProcessor:
    """Handles processing of the Yelp Open Dataset"""
    
    def __init__(self, business_file, review_file, output_dir="processed_data"):
        self.business_file = business_file
        self.review_file = review_file
        self.output_dir = output_dir
        self.businesses = {}
        self.categories_map = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def preprocess_text(self, text):
        """Preprocesses text using spaCy"""
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)
    
    def load_businesses(self):
        """Loads business data and creates a category mapping"""
        logger.info("Loading business data...")
        business_count = 0
        
        with open(self.business_file, "r", encoding="utf-8") as file:
            for line in tqdm(file):
                business = json.loads(line)
                business_id = business["business_id"]
                
                # Store business data
                self.businesses[business_id] = {
                    "name": business["name"],
                    "categories": business["categories"],
                    "city": business["city"],
                    "state": business["state"],
                    "address": business["address"],
                    "latitude": business["latitude"],
                    "longitude": business["longitude"],
                    "stars": business["stars"]
                }
                
                # Create category mapping
                if business["categories"]:
                    categories = [cat.strip() for cat in business["categories"].split(",")]
                    for category in categories:
                        if category not in self.categories_map:
                            self.categories_map[category] = []
                        self.categories_map[category].append(business_id)
                
                business_count += 1
        
        logger.info(f"Loaded {business_count} businesses with {len(self.categories_map)} unique categories")
        
    def process_reviews(self, limit=None):
        """Processes Yelp reviews and creates a dataset for analysis"""
        logger.info("Processing Yelp reviews...")
        reviews_data = []
        count = 0
        
        with open(self.review_file, "r", encoding="utf-8") as file:
            for line in tqdm(file):
                if limit and count >= limit:
                    break
                    
                review = json.loads(line)
                business_id = review["business_id"]
                
                if business_id in self.businesses:
                    # Get business categories
                    categories = self.businesses[business_id].get("categories", "")
                    
                    # Preprocess review text
                    cleaned_text = self.preprocess_text(review["text"])
                    
                    # Add to dataset
                    reviews_data.append({
                        "review_id": review["review_id"],
                        "business_id": business_id,
                        "user_id": review["user_id"],
                        "stars": review["stars"],
                        "text": review["text"],
                        "cleaned_text": cleaned_text,
                        "date": review["date"],
                        "categories": categories,
                        "source": "yelp"
                    })
                    
                    count += 1
        
        logger.info(f"Processed {count} Yelp reviews")
        return pd.DataFrame(reviews_data)


class ProfessionalReviewScraper:
    """Handles scraping of professional reviews from various sources"""
    
    def __init__(self, categories_map, output_dir="scraped_data"):
        self.categories_map = categories_map
        self.output_dir = output_dir
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Sources mapping (category to professional review sources)
        self.sources_mapping = {
            "Restaurants": ["tripadvisor.com/Restaurants", "michelin.com/restaurants", "eater.com", "theinfatuation.com"],
            "Hotels": ["tripadvisor.com/Hotels", "booking.com", "hotels.com"],
            "Nightlife": ["timeout.com", "thrillist.com"],
            "Shopping": ["trustpilot.com", "consumerreports.org"],
            "Bars": ["timeout.com/bars", "thrillist.com/drink"],
            "Coffee & Tea": ["perfectdailygrind.com", "sprudge.com"],
            "Beauty & Spas": ["allure.com", "spaindex.com"],
            "Home Services": ["homeadvisor.com", "angieslist.com"]
        }
        
        # Initialize datastreamer-like API client (simulated)
        self.datastreamer = DatastreamerClient()
    
    def get_sources_for_category(self, category):
        """Returns appropriate sources for a given category"""
        # Match partial category string
        for key in self.sources_mapping:
            if key in category:
                return self.sources_mapping[key]
        
        # Default sources
        return ["trustpilot.com", "yelp.com"]
    
    def normalize_business_name(self, name):
        """Normalizes business name for search queries"""
        return re.sub(r'[^\w\s]', '', name.lower()).replace(" ", "+")
    
    def scrape_professional_reviews(self, df_businesses, limit_per_source=5):
        """Scrapes professional reviews for businesses"""
        logger.info("Scraping professional reviews...")
        professional_reviews = []
        
        # Get unique businesses from the dataframe
        unique_businesses = []
        for business_id, business_data in tqdm(list(df_businesses.items())[:100]):  # Limit to 100 for demo
            unique_businesses.append({
                "id": business_id,
                "name": business_data["name"],
                "categories": business_data["categories"],
                "city": business_data["city"],
                "state": business_data["state"]
            })
        
        logger.info(f"Found {len(unique_businesses)} unique businesses to scrape")
        
        # Process each business
        for business in tqdm(unique_businesses):
            # Get categories
            if not business["categories"]:
                continue
                
            categories = [cat.strip() for cat in business["categories"].split(",")]
            
            # For each category, get appropriate sources
            for category in categories[:1]:  # Limit to first category for demo
                sources = self.get_sources_for_category(category)
                
                # Use the Datastreamer client instead of direct scraping when possible
                try:
                    datastreamer_reviews = self.datastreamer.get_reviews(
                        business_name=business["name"],
                        category=category,
                        location=f"{business['city']}, {business['state']}",
                        limit=limit_per_source
                    )
                    
                    for review in datastreamer_reviews:
                        review_id = hashlib.md5(f"{review['source']}_{review['date']}_{business['id']}".encode()).hexdigest()
                        professional_reviews.append({
                            "review_id": review_id,
                            "business_id": business["id"],
                            "business_name": business["name"],
                            "text": review["text"],
                            "stars": review["rating"],
                            "date": review["date"],
                            "source": review["source"],
                            "critic_name": review.get("author", ""),
                            "categories": business["categories"]
                        })
                
                except Exception as e:
                    logger.error(f"Error using Datastreamer for {business['name']}: {str(e)}")
                    
                    # Fallback to direct scraping for each source
                    for source in sources[:1]:  # Limit to first source for demo
                        try:
                            # Generate search query
                            query = f"{self.normalize_business_name(business['name'])}+{business['city']}+review+site:{source}"
                            reviews = self.scrape_source(query, business["id"], business["name"], limit_per_source)
                            professional_reviews.extend(reviews)
                        except Exception as e:
                            logger.error(f"Error scraping {source} for {business['name']}: {str(e)}")
        
        logger.info(f"Scraped {len(professional_reviews)} professional reviews")
        return pd.DataFrame(professional_reviews)
    
    def scrape_source(self, query, business_id, business_name, limit):
        """Simulates scraping from a specific source using a search query"""
        reviews = []
        
        # In a real implementation, this would do an actual web request and parsing
        # For simulation, we'll generate some realistic dummy data
        for i in range(limit):
            review_date = datetime.now().strftime("%Y-%m-%d")
            source = query.split("site:")[-1].split()[0]
            
            # Generate a unique review ID
            review_id = hashlib.md5(f"{source}_{review_date}_{business_id}_{i}".encode()).hexdigest()
            
            reviews.append({
                "review_id": review_id,
                "business_id": business_id,
                "business_name": business_name,
                "text": f"This is a simulated professional review for {business_name} from {source}. The food was excellent and service impeccable. Highly recommended for a fine dining experience.",
                "stars": 4.5,  # Simulated rating
                "date": review_date,
                "source": source,
                "critic_name": f"Critic {i+1}",
                "categories": "Restaurants, Food"  # Placeholder
            })
            
            # Add delay to simulate network request
            time.sleep(0.1)
            
        return reviews


class DatastreamerClient:
    """Simulates the Datastreamer API for accessing professional reviews"""
    
    def __init__(self):
        self.sources = ['tripadvisor', 'michelin_guide', 'trustpilot', 'timeout', 'eater']
        
    def get_reviews(self, business_name, category, location, limit=5):
        """Simulates fetching reviews from the Datastreamer API"""
        # This is a simulation of what the actual API would return
        reviews = []
        
        for i in range(limit):
            # Select a random source for variety
            source = self.sources[i % len(self.sources)]
            
            # Generate a realistic review
            if "Restaurant" in category or "Food" in category:
                text = f"[Professional Review] {business_name} offers an exceptional dining experience. The chef's special was outstanding, with perfect flavor balance and presentation. The atmosphere is elegant yet comfortable, and service is attentive without being intrusive. A must-visit establishment in {location.split(',')[0]}."
                rating = 4.5
            elif "Hotel" in category:
                text = f"[Professional Review] {business_name} provides excellent accommodations with attention to detail. Rooms are spacious and well-appointed, staff is friendly and efficient. The location in {location.split(',')[0]} offers convenient access to local attractions. Highly recommended for business or leisure travelers."
                rating = 4.2
            else:
                text = f"[Professional Review] {business_name} exceeds expectations in the {category} category. Located in {location.split(',')[0]}, it offers professional service and excellent value. The establishment stands out for its commitment to quality and customer satisfaction."
                rating = 4.0
            
            # Generate a date within the last year
            days_ago = (i * 30) % 365
            date_obj = datetime.now()
            date_str = date_obj.strftime("%Y-%m-%d")
            
            reviews.append({
                "text": text,
                "rating": rating,
                "date": date_str,
                "source": source,
                "author": f"Professional Critic {i+1}",
                "url": f"https://{source}.com/reviews/{business_name.lower().replace(' ', '-')}"
            })
        
        return reviews


class ReviewAligner:
    """Aligns professional reviews with Yelp reviews for comparison"""
    
    def __init__(self, yelp_df, professional_df):
        self.yelp_df = yelp_df
        self.professional_df = professional_df
        self.nlp = spacy.load("en_core_web_sm")
        
    def preprocess_text(self, text):
        """Preprocesses text for comparison"""
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)
    
    def create_doc2vec_model(self, combined_df):
        """Creates a Doc2Vec model for text similarity"""
        logger.info("Creating Doc2Vec model...")
        tagged_data = []
        
        for idx, row in combined_df.iterrows():
            # Preprocess text if not already preprocessed
            text = row.get("cleaned_text", self.preprocess_text(row["text"]))
            
            # Create tagged document
            tagged_data.append(TaggedDocument(words=text.split(), 
                                             tags=[f"{row['source']}_{row['review_id']}_{idx}"]))
        
        # Train Doc2Vec model
        model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=1, workers=4, epochs=20)
        logger.info("Doc2Vec model created")
        return model
    
    def align_reviews(self):
        """Aligns professional reviews with Yelp reviews based on business"""
        logger.info("Aligning professional reviews with Yelp reviews...")
        
        # Preprocess professional review text
        self.professional_df["cleaned_text"] = self.professional_df["text"].apply(self.preprocess_text)
        
        # Combine datasets for model training
        combined_df = pd.concat([
            self.yelp_df[["review_id", "business_id", "text", "cleaned_text", "source"]],
            self.professional_df[["review_id", "business_id", "text", "cleaned_text", "source"]]
        ])
        
        # Create Doc2Vec model
        model = self.create_doc2vec_model(combined_df)
        
        # Group by business_id
        alignment_results = []
        matched_businesses = 0
        total_yelp_reviews_used = 0
        total_prof_reviews_used = 0

        business_groups = self.yelp_df.groupby("business_id")
        
        for business_id, yelp_group in tqdm(list(business_groups)):
            # Get professional reviews for the same business
            prof_group = self.professional_df[self.professional_df["business_id"] == business_id]
            
            if len(prof_group) == 0:
                continue

            used_yelp_ids = set()

            matched_businesses += 1
            total_yelp_reviews_used += len(yelp_group)
            total_prof_reviews_used += len(prof_group)
                
            # For each professional review, find the most similar Yelp reviews
            for prof_idx, prof_row in prof_group.iterrows():
                prof_vector = model.infer_vector(prof_row["cleaned_text"].split())
                
                # Calculate similarity with all Yelp reviews for this business
                similarities = []
                for yelp_idx, yelp_row in yelp_group.iterrows():
                    if yelp_row["review_id"] in used_yelp_ids:
                        continue  # Skip already used Yelp reviews

                    yelp_vector = model.infer_vector(yelp_row["cleaned_text"].split())
                    similarity = model.dv.cosine_similarities(prof_vector, [yelp_vector])[0]
                    
                    similarities.append({
                        "prof_review_id": prof_row["review_id"],
                        "yelp_review_id": yelp_row["review_id"],
                        "business_id": business_id,
                        "similarity": float(similarity),
                        "prof_source": prof_row["source"],
                        "prof_stars": prof_row.get("stars", 0),
                        "yelp_stars": yelp_row["stars"]
                    })
                
                # Sort by similarity and keep top matches
                if similarities:
                    best_match = max(similarities, key=lambda x: x["similarity"])
                    used_yelp_ids.add(best_match["yelp_review_id"])
                    alignment_results.append(best_match)

        
        logger.info(f"Created {len(alignment_results)} review alignments")
        logger.info(f"Matched businesses: {matched_businesses}")
        logger.info(f"Total Yelp reviews used from matched businesses: {total_yelp_reviews_used}")
        logger.info(f"Total professional reviews used from matched businesses: {total_prof_reviews_used}")

        return pd.DataFrame(alignment_results)


class ReviewProcessor:
    """Main class that orchestrates the entire review processing pipeline"""
    
    def __init__(self, yelp_business_file, yelp_review_file, output_dir="processed_data"):
        self.yelp_business_file = yelp_business_file
        self.yelp_review_file = yelp_review_file
        self.output_dir = output_dir
        
    def run_pipeline(self, limit_reviews=1000, limit_professional=5):
        """Runs the complete review processing pipeline"""
        logger.info("Starting review processing pipeline...")
        
        # Step 1: Process Yelp data
        yelp_processor = YelpDataProcessor(self.yelp_business_file, self.yelp_review_file, self.output_dir)
        yelp_processor.load_businesses()
        yelp_reviews_df = yelp_processor.process_reviews(limit=limit_reviews)
        
        # Step 2: Scrape professional reviews
        scraper = ProfessionalReviewScraper(yelp_processor.categories_map, self.output_dir)
        professional_reviews_df = scraper.scrape_professional_reviews(
            yelp_processor.businesses, 
            limit_per_source=limit_professional
        )
        
        # Step 3: Align reviews
        aligner = ReviewAligner(yelp_reviews_df, professional_reviews_df)
        alignment_df = aligner.align_reviews()
        
        # Step 4: Save processed data
        self.save_results(yelp_reviews_df, professional_reviews_df, alignment_df)
        
        logger.info("Review processing pipeline completed successfully")
        return {
            "yelp_reviews": yelp_reviews_df,
            "professional_reviews": professional_reviews_df,
            "alignments": alignment_df
        }
    
    def save_results(self, yelp_df, professional_df, alignment_df):
        """Saves all processed data to files"""
        logger.info("Saving processed data...")
        
        # Save to CSV files
        yelp_df.to_csv(os.path.join(self.output_dir, "yelp_reviews_processed.csv"), index=False)
        professional_df.to_csv(os.path.join(self.output_dir, "professional_reviews.csv"), index=False)
        alignment_df.to_csv(os.path.join(self.output_dir, "review_alignments.csv"), index=False)
        
        # Save sample to JSON for inspection
        yelp_sample = yelp_df.head(10).to_dict(orient="records")
        prof_sample = professional_df.head(10).to_dict(orient="records")
        alignment_sample = alignment_df.head(10).to_dict(orient="records")
        
        with open(os.path.join(self.output_dir, "samples.json"), "w") as f:
            json.dump({
                "yelp_sample": yelp_sample,
                "professional_sample": prof_sample,
                "alignment_sample": alignment_sample
            }, f, indent=2)
        
        logger.info(f"Data saved to {self.output_dir}")




if __name__ == "__main__":
    # Configuration
    YELP_BUSINESS_FILE = "yelp_academic_dataset_business.json"
    YELP_REVIEW_FILE = "yelp_academic_dataset_review.json"
    OUTPUT_DIR = "processed_reviews"
    
    # Run the pipeline with limited data for testing
    processor = ReviewProcessor(YELP_BUSINESS_FILE, YELP_REVIEW_FILE, OUTPUT_DIR)
    results = processor.run_pipeline(limit_reviews=5000, limit_professional=3)
    
    # Print summary
    print(f"Processed {len(results['yelp_reviews'])} Yelp reviews")
    print(f"Collected {len(results['professional_reviews'])} professional reviews")
    print(f"Created {len(results['alignments'])} review alignments")