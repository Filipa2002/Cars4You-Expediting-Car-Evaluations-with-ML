import pandas as pd # type:ignore
from typing import Dict, Any, Optional, Union, Callable, List
from rapidfuzz import fuzz # type:ignore 


class CardinalMapper:
    def __init__(self, mapping_dict: Dict[str, List[str]], uncertain_threshold: int = 50):
        self.mapping_dict = mapping_dict
        self.uncertain_threshold = uncertain_threshold

    def fit(self, X: pd.Series, y: Optional[pd.Series] = None) -> 'CardinalMapper':
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        unique_values = X.unique()
        # create the mapping table
        mapping_table = {}
        for value in unique_values:
            mapping_table[value] = self._choose_cardinal_internal(value)
        return X.map(mapping_table)
    

    def fit_transform(self, X: pd.Series, y: Optional[pd.Series] = None) -> pd.Series:
        self.fit(X, y)
        return self.transform(X)
    
    def _choose_cardinal_internal(self, word: str) -> str:
        try:
            word = word.lower().strip()
        except:
            return 'Other' # if word is not a string it's probably nan 
        for cardinal, (valid_variations, false_friends) in self.mapping_dict.items():
            # Calculate scores for valid variations
            valid_scores = [fuzz.ratio(word.lower(), variation.lower()) for variation in valid_variations]
            false_scores = [fuzz.ratio(word.lower(), false_friend.lower()) for false_friend in false_friends]
            
            top_valid_scores = sorted(valid_scores, reverse=True)[:3]
            top_false_scores = sorted(false_scores, reverse=True)[:3]
            
            score_difference = sum(top_valid_scores) - sum(top_false_scores)
            
            if score_difference > self.uncertain_threshold:  # Threshold to consider a match
                return cardinal
        return 'Other'