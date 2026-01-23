import pandas as pd
from typing import Dict, Optional
from loguru import logger


class HCCLookupService:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = self._load_csv()

    def _load_csv(self) -> pd.DataFrame:
        logger.info(f"Loading HCC CSV: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        df["condition_normalized"] = df["Description"].str.lower().str.strip()
        return df

    def lookup(self, condition: str) -> Optional[Dict[str, str]]:
        key = condition.lower().strip()
        
        # First try exact match
        match = self.df[self.df["condition_normalized"] == key]
        
        # If no exact match, try substring/fuzzy matching
        if match.empty:
            # Check if the condition name is contained in any description
            match = self.df[self.df["condition_normalized"].str.contains(key, case=False, na=False)]
        
        # If still no match, try the reverse - check if any description is contained in the condition
        if match.empty:
            match = self.df[self.df["condition_normalized"].apply(
                lambda x: x in key if pd.notna(x) else False
            )]
        
        # If still no match, try word-based matching (split and match key words)
        if match.empty:
            key_words = set(key.split())
            if len(key_words) > 0:
                # Find rows where at least 2 key words match, and score by number of matching words
                def score_match(desc):
                    if pd.isna(desc):
                        return 0
                    desc_words = set(str(desc).lower().split())
                    return len(key_words.intersection(desc_words))
                
                scored = self.df.copy()
                scored['_match_score'] = scored["condition_normalized"].apply(score_match)
                match = scored[scored['_match_score'] >= min(2, len(key_words))]
                
                # Sort by match score (descending) and take the best match
                if not match.empty:
                    match = match.sort_values('_match_score', ascending=False)
        
        if match.empty:
            logger.debug(f"No match found for condition: {condition}")
            return None

        # Take the first (best) match
        row = match.iloc[0]
        
        # Get column values - handle potential column name variations
        code_col = "ICD-10-CM Codes"
        desc_col = "Description"
        tags_col = "Tags"
        
        # Verify columns exist
        if code_col not in row.index:
            logger.warning(f"Column '{code_col}' not found in CSV. Available columns: {list(self.df.columns)}")
            return None
        
        # Get code - handle NaN, None, and empty strings
        code_value = row[code_col]
        if pd.isna(code_value) or code_value is None:
            code = None
        else:
            code = str(code_value).strip()
            if code == "" or code == "nan":
                code = None
        
        # Get description
        description = str(row[desc_col]).strip() if pd.notna(row[desc_col]) else condition
        
        # Check if Tags column indicates HCC relevance (non-empty tag means HCC relevant)
        hcc_relevant = False
        if tags_col in row.index:
            hcc_relevant = pd.notna(row[tags_col]) and str(row[tags_col]).strip() != ""
        
        logger.debug(f"Matched '{condition}' to '{description}' with code '{code}'")
        
        return {
            "condition": description,
            "code": code,
            "hcc_relevant": hcc_relevant,
        }
