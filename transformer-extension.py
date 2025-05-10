import os
import json
import spacy
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Initialize logging
tog = logging.getLogger("SBERTReviewAligner")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load spaCy model for text preprocessing
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    """Lowercase, lemmatize, remove stopwords and non-alpha tokens."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)


def get_season(date_str: str) -> str:
    """Return meteorological season for a given YYYY-MM-DD date."""
    month = int(date_str.split('-')[1])
    if month in (12, 1, 2): return 'winter'
    if month in (3, 4, 5):  return 'spring'
    if month in (6, 7, 8):  return 'summer'
    return 'fall'


class SBERTReviewAligner:
    """
    Aligns professional and Yelp reviews using transformer embeddings
    and applies adaptive similarity thresholds by category, region, and season.
    """
    def __init__(self,
                 yelp_df: pd.DataFrame,
                 prof_df: pd.DataFrame,
                 business_meta: pd.DataFrame,
                 threshold_config: dict):
        self.yelp_df = yelp_df.copy()
        self.prof_df = prof_df.copy()
        self.business_meta = business_meta.set_index('business_id')
        self.threshold_config = threshold_config
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Preprocess and embed texts
        tog.info("Preprocessing and embedding Yelp reviews...")
        self.yelp_df['cleaned'] = self.yelp_df['text'].apply(preprocess_text)
        y_texts = self.yelp_df['cleaned'].tolist()
        self.yelp_df['embeddings'] = list(self.model.encode(y_texts, show_progress_bar=False))

        tog.info("Preprocessing and embedding professional reviews...")
        self.prof_df['cleaned'] = self.prof_df['text'].apply(preprocess_text)
        p_texts = self.prof_df['cleaned'].tolist()
        self.prof_df['embeddings'] = list(self.model.encode(p_texts, show_progress_bar=False))

    def align_and_filter(self) -> pd.DataFrame:
        """
        For each professional review, find the best-matching Yelp review
        and filter by an adaptive threshold.
        """
        records = []
        # Group Yelp reviews by business
        for b_id, group_y in self.yelp_df.groupby('business_id'):
            group_p = self.prof_df[self.prof_df['business_id'] == b_id]
            if group_p.empty:
                continue
            # Retrieve region and categories from metadata
            meta = self.business_meta.loc[b_id]
            region = meta.get('state', None)
            cats = meta.get('categories', '')

            y_embs = np.vstack(group_y['embeddings'].values)

            for _, prow in group_p.iterrows():
                p_emb = prow['embeddings']
                # cosine similarity matrix
                sims = util.cos_sim(p_emb, y_embs)[0].cpu().numpy()
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                yrow = group_y.iloc[best_idx]

                # compute dynamic threshold
                threshold = self._get_threshold(region, cats, prow['date'])
                if best_sim >= threshold:
                    records.append({
                        'business_id': b_id,
                        'prof_review_id': prow['review_id'],
                        'yelp_review_id': yrow['review_id'],
                        'similarity': best_sim
                    })

        return pd.DataFrame(records)

    def _get_threshold(self, region, categories_str: str, date_str: str) -> float:
        """
        Determine threshold based on region, category, and season.
        Takes the maximum of applicable thresholds.
        """
        default = self.threshold_config.get('default', 0.5)
        # Season
        season = get_season(date_str)
        season_thr = self.threshold_config.get('seasonal', {}).get(season, default)
        # Category: assume comma-separated list
        cat_thr = default
        for cat in categories_str.split(','):
            cat = cat.strip()
            if cat in self.threshold_config.get('category', {}):
                cat_thr = self.threshold_config['category'][cat]
                break
        # Region (state)
        region_thr = self.threshold_config.get('region', {}).get(region, default)

        return max(default, season_thr, cat_thr, region_thr)


# Example usage:
# from review_processor_extension import SBERTReviewAligner
# yelp_df = pd.read_csv('processed_data/yelp_reviews_processed.csv')
# prof_df = pd.read_csv('processed_data/professional_reviews.csv')
# business_meta = pd.DataFrame.from_dict(yelp_processor.businesses, orient='index').reset_index().rename(columns={'index': 'business_id'})
# thresholds = {'default':0.5,
#               'seasonal':{'summer':0.6,'winter':0.55},
#               'category':{'Restaurants':0.65},
#               'region':{'CA':0.6,'NY':0.62}}
# aligner = SBERTReviewAligner(yelp_df, prof_df, business_meta, thresholds)
# filtered_alignments = aligner.align_and_filter()
# filtered_alignments.to_csv('filtered_alignments.csv', index=False)
