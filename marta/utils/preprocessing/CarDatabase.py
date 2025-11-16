
#! pip install / conda install rapidfuzz
#! pip install kagglehub
from typing import List, Dict, Tuple

import kagglehub # type:ignore 
import os
import pandas as pd # type:ignore 
import json
from rapidfuzz import fuzz # type:ignore 
import random
import json
import re
from collections import defaultdict, Counter
from tqdm import tqdm  # type:ignore

def create_optimized_database(car_data:List[Dict], max_model_words: int = 2) -> Dict:
    """
    Create an optimized database with frequency-based simplification and manual overrides
    
    Args:
        car_data (List[Dict]): Original car database with 'Make' and 'Models'
        max_model_words (int): Maximum number of words to keep in simplified model names
    Returns:
        Dict: Simplified car database with canonical names, aliases, and simplified models
    """
    simplified_db = {}
    
    # Enhanced brand aliases added for better normalization
    brand_aliases = {
        'volkswagen': ['vw','volks', 'volkswagon'],
        'mercedes-benz': ['mercedes', 'merce', 'merc', 'mb', 'mercedesbenz'],
        'bmw': ['beemer', 'bimmer'],
        'toyota': ['toyo'],
        'hyundai': ['hundai', 'hyunday'],
        'chevrolet': ['chevy', 'chev'],
        'cadillac': ['caddy'],
        'infiniti': ['infinity'],
        'peugeot': ['peugot', 'pugeot'],
        'mitsubishi': ['mitsu', 'mitsubisi']
    }
    
    brand_to_canonical = {}
    for canonical, aliases in brand_aliases.items():
        brand_to_canonical[canonical] = canonical
        for alias in aliases:
            brand_to_canonical[alias] = canonical
    
    def normalize_text(text):
        """Normalize text for comparison"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    # Analyze word frequencies for intelligent simplification
    print("Analyzing word frequencies...")
    all_words = []
    for item in car_data:
        for model in item['Models']:
            words = normalize_text(model).split()
            all_words.extend(words)
    
    word_freq = Counter(all_words)
    total_words = len(all_words)
    
    # Create word importance scores
    word_importance = {}
    for word, freq in word_freq.items():
        relative_freq = freq / total_words
        
        # High frequency or very low frequency words are less important
        if relative_freq > 0.05:
            importance = 0.1
        elif relative_freq < 0.0001:
            importance = 0.2
        else:
            importance = 1.0
        
        # Automotive-specific rules
        if word in ['sedan', 'coupe', 'wagon', 'hatchback', 'convertible', 'suv', 'cabriolet']:
            importance = 0.1
        elif word in ['automatic', 'manual', 'cvt', 'awd', '4wd', 'fwd', 'rwd']:
            importance = 0.1
        elif re.match(r'\d+\.\d+l?', word):  # Engine sizes
            importance = 0.1
        elif word in ['turbo', 'hybrid', 'electric', 'diesel', 'petrol']:
            importance = 0.2
        elif re.match(r'^(19|20)\d{2}$', word):  # Years
            importance = 0.1
        
        word_importance[word] = importance
    
    def simplify_model_name(words: List[str], max_words:int=max_model_words) -> str:
        """
            Simplify model name by keeping most important words based on frequency analysis
            
            Args:
                words (List[str]): List of words in the model name
                max_words (int): Maximum number of words to keep
            Returns:
                str: Simplified model name
        """
        if len(words) <= max_words:
            return ' '.join(words)
        
        # Score words by importance
        word_scores = [(word, word_importance.get(word, 0.5)) for word in words]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        important_words = [word for word, score in word_scores[:max_words]]
        
        # Maintain original order
        simplified_words = []
        for word in words:
            if word in important_words:
                simplified_words.append(word)
                if len(simplified_words) == max_words:
                    break
        
        return ' '.join(simplified_words)
    
    # Process database
    for item in car_data:
        canonical_make = normalize_text(item['Make'])
        
        if canonical_make in brand_to_canonical:
            canonical_make = brand_to_canonical[canonical_make]
        
        if canonical_make not in simplified_db:
            simplified_db[canonical_make] = {
                'canonical_name': canonical_make,
                'aliases': brand_aliases.get(canonical_make, []),
                'models': set()
            }
        
        # Process models
        for model in item['Models']:
            words = normalize_text(model).split()
            simplified_model = simplify_model_name(words, max_model_words)
            
            if simplified_model and len(simplified_model) > 2:
                simplified_db[canonical_make]['models'].add(simplified_model)
    
    # Manual additions for missing popular models
    manual_additions = {
        'audi': ['q2', 'q4', 'q7', 'q8',
                 'a1', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8',
                 'tt', 'r8'],
        'mercedes-benz': ['a class', 'b class', 'c class', 'e class', 'g class', 's class','x class','cla class','cls class',
                         'glc class', 'gla class', 'gle class', 'gls class'],
        'bmw': ['1 series', '2 series', '4 series', '5 series', '6 series', '7 series', '8 series'],
        'volkswagen': ['id3', 'id4', 'id buzz', 'tiguan', 'touareg', 'arteon', 'polo', 'golf', 'passat', 'touran', 'sharan', 'up'],
        'toyota': ['yaris', 'chr', 'rav4', 'highlander', 'supra', 'c hr', 'aygo'],
        'hyundai': ['i10', 'i20', 'i30', 'kona', 'tucson', 'santa fe', 'palisaide'],
        'skoda': ['kamiq', 'karoq', 'kodiaq', 'scala', 'octavia', 'superb', 'fabia', 'citigo'],
        'opel': ['corsa', 'astra', 'insignia', 'mokka', 'grandland', 'crossland', 'zafira', 'combo', 'vivaro', 'cascada', 'Meriva', 'adam', 'calibra']
    }
    
    for make, models in manual_additions.items():
        if make in simplified_db:
            simplified_db[make]['models'].update(models)
        else:
            simplified_db[make] = {
                'canonical_name': make,
                'aliases': brand_aliases.get(make, []),
                'models': set(models)
            }
    
    # Convert sets to sorted lists
    for make_data in simplified_db.values():
        make_data['models'] = sorted(list(make_data['models']))
    
    print(f"Created optimized database with {len(simplified_db)} makes")
    return simplified_db




class ProductionCarMatcher:
    """
    Production-ready car matcher with enhanced numeric matching, 
    strong brand constraints, and multiple fallback strategies
    """
    
    def __init__(self, simplified_db: Dict[str, List[str]]):
        self.db = simplified_db
        self.make_lookup = self._create_make_lookup()
        
    def _create_make_lookup(self):
        """Create lookup table for makes including aliases"""
        lookup = {}
        for canonical_make, data in self.db.items():
            lookup[canonical_make] = canonical_make
            for alias in data['aliases']:
                lookup[alias] = canonical_make
        return lookup
    
    def _normalize_input(self, text: str) -> str:
        """Normalize input text"""
        if not text or pd.isna(text):
            return ""
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def find_best_make_match(self, input_make: str, threshold:int=70) -> tuple[str, int]:
        """Find best make match using fuzzy matching"""
        if not input_make:
            return None, 0
        
        input_make = self._normalize_input(input_make)
        
        # Exact lookup first (including aliases)
        if input_make in self.make_lookup:
            return self.make_lookup[input_make], 100
        
        # Fuzzy matching
        best_match = None
        best_score = 0
        
        for make_option in self.make_lookup.keys():
            scores = [
                fuzz.ratio(input_make, make_option),
                fuzz.partial_ratio(input_make, make_option),
                fuzz.token_sort_ratio(input_make, make_option)
            ]
            score = max(scores)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = self.make_lookup[make_option]
        
        return best_match, best_score
    
    def find_best_model_match(self, input_model: str, make:str=None, threshold:float=70.0) -> Tuple[str, float, str]:
        """Find best model match with enhanced numeric matching"""
        if not input_model:
            return None, 0, None
        
        input_model = self._normalize_input(input_model)
        best_match = None
        best_score = 0
        best_make = None
        
        # Extract numbers for enhanced matching
        input_numbers = re.findall(r'\d+', input_model)
        
        # Search scope
        if make and make in self.db:
            search_scope = [(make, self.db[make]['models'])]
        else:
            search_scope = [(m, data['models']) for m, data in self.db.items()]
        
        for make_name, models in search_scope:
            for model in models:
                model_numbers = re.findall(r'\d+', model)
                
                # Calculate base similarity scores
                ratio_score = fuzz.ratio(input_model, model)
                partial_score = fuzz.partial_ratio(input_model, model)
                token_sort_score = fuzz.token_sort_ratio(input_model, model)
                
                # Prefer exact matches over partial matches
                if ratio_score >= 90:
                    score = ratio_score
                elif token_sort_score >= 90:
                    score = token_sort_score
                else:
                    score = max(ratio_score, token_sort_score, partial_score * 0.8)
                
                # Enhanced numeric matching
                numeric_bonus = 0
                if input_numbers and model_numbers:
                    matching_numbers = set(input_numbers) & set(model_numbers)
                    if matching_numbers:
                        # Bonus for matching numbers
                        numeric_bonus = len(matching_numbers) * 8
                        if set(input_numbers) == set(model_numbers):
                            numeric_bonus += 12
                    elif len(input_numbers) > 0 and len(model_numbers) > 0:
                        # Penalty for conflicting numbers
                        score *= 0.6
                
                # Word matching bonus
                input_words = set(input_model.split())
                model_words = set(model.split())
                if input_words.issubset(model_words) or model_words.issubset(input_words):
                    score += 5
                
                # Length penalty for very different lengths
                length_diff = abs(len(input_model) - len(model))
                if length_diff > 3:
                    score *= 0.95
                
                # Strong cross-brand penalty
                if make and make_name != make:
                    score *= 0.4
                    numeric_bonus *= 0.3
                
                final_score = min(score + numeric_bonus, 100)  # Cap at 100
                
                if final_score > best_score and final_score >= threshold:
                    best_score = final_score
                    best_match = model
                    best_make = make_name
        
        return best_match, best_score, best_make
    
    def clean_make_model_pair(self, input_make: str, input_model: str,
                            make_threshold:float=70.0, model_threshold=70.0) -> Dict:
        """Clean a make-model pair with multiple fallback strategies"""
        
        # Find best make match
        clean_make, make_score = self.find_best_make_match(input_make, make_threshold)
        
        # Strategy 1: Find model within the identified make
        clean_model, model_score, model_make = self.find_best_model_match(
            input_model, clean_make, model_threshold
        )
        
        # Strategy 2: If no model found, try with lower threshold within make
        if not clean_model and clean_make:
            clean_model, model_score, model_make = self.find_best_model_match(
                input_model, clean_make, model_threshold - 10
            )
        
        # Strategy 3: If still no model, try global search with higher threshold
        if not clean_model:
            clean_model, model_score, model_make = self.find_best_model_match(
                input_model, None, model_threshold + 10
            )
            # Update make if we found a model in different make
            if clean_model and model_make:
                clean_make = model_make
                make_score = self.find_best_make_match(model_make)[1]
        
        return {
            'original_make': input_make,
            'original_model': input_model,
            'clean_make': clean_make,
            'clean_model': clean_model,
            'make_score': make_score,
            'model_score': model_score,
            'confidence': (make_score + model_score) / 2 if make_score and model_score else 0
        }
    
    def clean_dataframe(self, df: pd.DataFrame, make_col:str='make', model_col:str='model') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean car make and model columns in a DataFrame"""
        results = []
        
        print(f"Cleaning {len(df)} records...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
            make_input = row[make_col] if make_col in df.columns else None
            model_input = row[model_col] if model_col in df.columns else None
            
            result = self.clean_make_model_pair(make_input, model_input)
            results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Add cleaned columns to original DataFrame
        df_cleaned = df.copy()
        df_cleaned[f'{make_col}_clean'] = results_df['clean_make']
        df_cleaned[f'{model_col}_clean'] = results_df['clean_model']
        df_cleaned[f'{make_col}_confidence'] = results_df['make_score']
        df_cleaned[f'{model_col}_confidence'] = results_df['model_score']
        df_cleaned['overall_confidence'] = results_df['confidence']
        
        return df_cleaned, results_df
    
    def get_cleaning_stats(self, results_df: pd.DataFrame) -> Dict:
        """Get statistics about the cleaning process"""
        stats = {
            'total_records': len(results_df),
            'successful_make_matches': sum(results_df['clean_make'].notna()),
            'successful_model_matches': sum(results_df['clean_model'].notna()),
            'high_confidence_matches': sum(results_df['confidence'] >= 90),
            'medium_confidence_matches': sum((results_df['confidence'] >= 70) & (results_df['confidence'] < 90)),
            'low_confidence_matches': sum(results_df['confidence'] < 70),
            'avg_confidence': results_df['confidence'].mean()
        }
        return stats

