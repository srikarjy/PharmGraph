"""Journal impact scoring system with domain-specific weighting for pharmacogenomics."""

import re
import math
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from ..config import get_logger


logger = get_logger(__name__)


class JournalTier(Enum):
    """Journal tier classifications."""
    TIER_1 = 1  # 95-100 points: Nature, Science, Cell, NEJM
    TIER_2 = 2  # 85-94 points: Nature family, high-impact specialty
    TIER_3 = 3  # 75-84 points: Top domain-specific journals
    TIER_4 = 4  # 65-74 points: Good domain journals, bioinformatics
    TIER_5 = 5  # 50-64 points: Specialty journals
    TIER_6 = 6  # 25-49 points: General/open access journals


@dataclass
class JournalInfo:
    """Journal information with scoring metadata."""
    name: str
    normalized_name: str
    tier: JournalTier
    base_score: float
    impact_factor: Optional[float] = None
    publisher: Optional[str] = None
    domain_relevance: float = 1.0  # Multiplier for pharmacogenomics relevance
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class JournalImpactScorer:
    """Comprehensive journal impact scoring system for pharmacogenomics research."""
    
    def __init__(self):
        """Initialize the journal scorer with comprehensive database."""
        self.journal_database: Dict[str, JournalInfo] = {}
        self.publisher_prestige: Dict[str, float] = {}
        
        # Initialize databases
        self._initialize_journal_database()
        self._initialize_publisher_prestige()
        
        logger.info(f"Initialized JournalImpactScorer with {len(self.journal_database)} journals")
    
    def _initialize_journal_database(self):
        """Initialize comprehensive journal database with impact factors and scoring."""
        
        # Tier 1: Top-tier general journals (95-100 points)
        tier1_journals = [
            JournalInfo("Nature", "nature", JournalTier.TIER_1, 100.0, 64.8, "Nature Publishing Group", 1.2),
            JournalInfo("Science", "science", JournalTier.TIER_1, 99.0, 63.7, "AAAS", 1.2),
            JournalInfo("Cell", "cell", JournalTier.TIER_1, 98.0, 64.5, "Cell Press", 1.3),
            JournalInfo("New England Journal of Medicine", "new england journal of medicine", JournalTier.TIER_1, 100.0, 176.1, "Massachusetts Medical Society", 1.1, 
                       ["NEJM", "N Engl J Med", "New Engl J Med"]),
            JournalInfo("The Lancet", "the lancet", JournalTier.TIER_1, 98.0, 168.9, "Elsevier", 1.1, ["Lancet"]),
            JournalInfo("JAMA", "jama", JournalTier.TIER_1, 96.0, 157.3, "American Medical Association", 1.1,
                       ["Journal of the American Medical Association"]),
        ]
        
        # Tier 2: High-impact specialty journals (85-94 points)
        tier2_journals = [
            JournalInfo("Nature Genetics", "nature genetics", JournalTier.TIER_2, 95.0, 31.7, "Nature Publishing Group", 1.5,
                       ["Nat Genet", "Nat. Genet."]),
            JournalInfo("Nature Medicine", "nature medicine", JournalTier.TIER_2, 94.0, 82.9, "Nature Publishing Group", 1.3,
                       ["Nat Med", "Nat. Med."]),
            JournalInfo("Nature Biotechnology", "nature biotechnology", JournalTier.TIER_2, 93.0, 68.2, "Nature Publishing Group", 1.4,
                       ["Nat Biotechnol", "Nat. Biotechnol."]),
            JournalInfo("Nature Machine Intelligence", "nature machine intelligence", JournalTier.TIER_2, 92.0, 25.9, "Nature Publishing Group", 1.4,
                       ["Nat Mach Intell", "Nat. Mach. Intell."]),
            JournalInfo("Science Translational Medicine", "science translational medicine", JournalTier.TIER_2, 91.0, 19.3, "AAAS", 1.3,
                       ["Sci Transl Med", "Sci. Transl. Med."]),
            JournalInfo("Cell Metabolism", "cell metabolism", JournalTier.TIER_2, 90.0, 31.3, "Cell Press", 1.2),
            JournalInfo("Circulation", "circulation", JournalTier.TIER_2, 88.0, 37.8, "American Heart Association", 1.2),
            JournalInfo("Blood", "blood", JournalTier.TIER_2, 87.0, 25.5, "American Society of Hematology", 1.1),
        ]
        
        # Tier 3: Top pharmacogenomics and domain-specific journals (75-84 points)
        tier3_journals = [
            JournalInfo("Clinical Pharmacology & Therapeutics", "clinical pharmacology & therapeutics", JournalTier.TIER_3, 90.0, 6.8, "Wiley", 1.5,
                       ["Clin Pharmacol Ther", "CPT", "Clinical Pharmacology and Therapeutics"]),
            JournalInfo("Pharmacogenomics", "pharmacogenomics", JournalTier.TIER_3, 85.0, 3.1, "Future Medicine", 1.5),
            JournalInfo("Genome Medicine", "genome medicine", JournalTier.TIER_3, 82.0, 5.4, "BioMed Central", 1.4),
            JournalInfo("Pharmacogenetics and Genomics", "pharmacogenetics and genomics", JournalTier.TIER_3, 80.0, 2.9, "Lippincott Williams & Wilkins", 1.5,
                       ["Pharmacogenet Genomics"]),
            JournalInfo("Human Molecular Genetics", "human molecular genetics", JournalTier.TIER_3, 78.0, 3.1, "Oxford University Press", 1.3,
                       ["Hum Mol Genet"]),
            JournalInfo("American Journal of Human Genetics", "american journal of human genetics", JournalTier.TIER_3, 85.0, 12.0, "Cell Press", 1.4,
                       ["Am J Hum Genet", "AJHG"]),
            JournalInfo("Nature Reviews Drug Discovery", "nature reviews drug discovery", JournalTier.TIER_3, 88.0, 112.3, "Nature Publishing Group", 1.4,
                       ["Nat Rev Drug Discov"]),
            JournalInfo("Drug Metabolism and Disposition", "drug metabolism and disposition", JournalTier.TIER_3, 76.0, 4.9, "ASPET", 1.4,
                       ["Drug Metab Dispos"]),
        ]
        
        # Tier 4: Good domain journals and bioinformatics (65-74 points)
        tier4_journals = [
            JournalInfo("Bioinformatics", "bioinformatics", JournalTier.TIER_4, 75.0, 5.8, "Oxford University Press", 1.3),
            JournalInfo("Journal of the American Medical Informatics Association", "journal of the american medical informatics association", JournalTier.TIER_4, 72.0, 4.7, "Oxford University Press", 1.2,
                       ["JAMIA", "J Am Med Inform Assoc"]),
            JournalInfo("Journal of Biomedical Informatics", "journal of biomedical informatics", JournalTier.TIER_4, 70.0, 4.0, "Elsevier", 1.2,
                       ["J Biomed Inform"]),
            JournalInfo("Pharmacotherapy", "pharmacotherapy", JournalTier.TIER_4, 68.0, 3.3, "Wiley", 1.3),
            JournalInfo("European Journal of Clinical Pharmacology", "european journal of clinical pharmacology", JournalTier.TIER_4, 67.0, 3.8, "Springer", 1.3,
                       ["Eur J Clin Pharmacol"]),
            JournalInfo("British Journal of Clinical Pharmacology", "british journal of clinical pharmacology", JournalTier.TIER_4, 69.0, 4.2, "Wiley", 1.3,
                       ["Br J Clin Pharmacol"]),
            JournalInfo("Clinical Pharmacokinetics", "clinical pharmacokinetics", JournalTier.TIER_4, 71.0, 4.6, "Springer", 1.3,
                       ["Clin Pharmacokinet"]),
            JournalInfo("Therapeutic Drug Monitoring", "therapeutic drug monitoring", JournalTier.TIER_4, 66.0, 3.2, "Lippincott Williams & Wilkins", 1.3,
                       ["Ther Drug Monit"]),
        ]
        
        # Tier 5: Specialty journals (50-64 points)
        tier5_journals = [
            JournalInfo("Personalized Medicine", "personalized medicine", JournalTier.TIER_5, 62.0, 2.1, "Future Medicine", 1.4),
            JournalInfo("The Pharmacogenomics Journal", "the pharmacogenomics journal", JournalTier.TIER_5, 64.0, 2.8, "Nature Publishing Group", 1.5,
                       ["Pharmacogenomics J"]),
            JournalInfo("Current Drug Metabolism", "current drug metabolism", JournalTier.TIER_5, 58.0, 2.3, "Bentham Science", 1.3),
            JournalInfo("Drug Metabolism Reviews", "drug metabolism reviews", JournalTier.TIER_5, 60.0, 4.3, "Taylor & Francis", 1.3,
                       ["Drug Metab Rev"]),
            JournalInfo("Xenobiotica", "xenobiotica", JournalTier.TIER_5, 55.0, 2.1, "Taylor & Francis", 1.2),
            JournalInfo("Journal of Clinical Pharmacy and Therapeutics", "journal of clinical pharmacy and therapeutics", JournalTier.TIER_5, 54.0, 2.8, "Wiley", 1.2,
                       ["J Clin Pharm Ther"]),
            JournalInfo("Pharmacogenomics and Personalized Medicine", "pharmacogenomics and personalized medicine", JournalTier.TIER_5, 56.0, 1.8, "Dove Medical Press", 1.4),
            JournalInfo("Current Pharmacogenomics and Personalized Medicine", "current pharmacogenomics and personalized medicine", JournalTier.TIER_5, 52.0, 1.5, "Bentham Science", 1.4),
        ]
        
        # Tier 6: General and open access journals (25-49 points)
        tier6_journals = [
            JournalInfo("PLoS ONE", "plos one", JournalTier.TIER_6, 45.0, 3.7, "PLOS", 1.0),
            JournalInfo("Scientific Reports", "scientific reports", JournalTier.TIER_6, 48.0, 4.6, "Nature Publishing Group", 1.0,
                       ["Sci Rep"]),
            JournalInfo("BMC Medical Genomics", "bmc medical genomics", JournalTier.TIER_6, 42.0, 4.0, "BioMed Central", 1.2),
            JournalInfo("BMC Bioinformatics", "bmc bioinformatics", JournalTier.TIER_6, 44.0, 3.3, "BioMed Central", 1.1),
            JournalInfo("Frontiers in Pharmacology", "frontiers in pharmacology", JournalTier.TIER_6, 40.0, 5.6, "Frontiers Media", 1.2,
                       ["Front Pharmacol"]),
            JournalInfo("Frontiers in Genetics", "frontiers in genetics", JournalTier.TIER_6, 38.0, 4.8, "Frontiers Media", 1.1,
                       ["Front Genet"]),
            JournalInfo("International Journal of Molecular Sciences", "international journal of molecular sciences", JournalTier.TIER_6, 35.0, 6.2, "MDPI", 1.0,
                       ["Int J Mol Sci"]),
            JournalInfo("Genes", "genes", JournalTier.TIER_6, 32.0, 4.1, "MDPI", 1.1),
            JournalInfo("Journal of Personalized Medicine", "journal of personalized medicine", JournalTier.TIER_6, 46.0, 4.9, "MDPI", 1.3,
                       ["J Pers Med"]),
        ]
        
        # Combine all journals
        all_journals = tier1_journals + tier2_journals + tier3_journals + tier4_journals + tier5_journals + tier6_journals
        
        # Build database with normalized names and aliases
        for journal in all_journals:
            # Add main entry
            self.journal_database[journal.normalized_name] = journal
            
            # Add aliases
            for alias in journal.aliases:
                normalized_alias = self.normalize_journal_name(alias)
                self.journal_database[normalized_alias] = journal
    
    def _initialize_publisher_prestige(self):
        """Initialize publisher prestige bonuses."""
        self.publisher_prestige = {
            "Nature Publishing Group": 5.0,
            "AAAS": 5.0,  # Science family
            "Cell Press": 5.0,
            "Massachusetts Medical Society": 4.0,  # NEJM
            "Elsevier": 2.0,
            "Wiley": 2.0,
            "Oxford University Press": 3.0,
            "Springer": 2.0,
            "Future Medicine": 1.0,
            "BioMed Central": 1.0,
            "PLOS": 1.0,
            "Frontiers Media": 0.5,
            "MDPI": 0.0,
            "Dove Medical Press": 0.5,
            "Bentham Science": 0.0,
        }
    
    def normalize_journal_name(self, journal_name: str) -> str:
        """Normalize journal name for consistent matching.
        
        Args:
            journal_name: Raw journal name
            
        Returns:
            str: Normalized journal name
        """
        if not journal_name:
            return ""
        
        # Convert to lowercase
        normalized = journal_name.lower().strip()
        
        # Remove common prefixes/suffixes
        normalized = re.sub(r'^the\s+', '', normalized)
        normalized = re.sub(r'\s+journal$', '', normalized)
        
        # Standardize punctuation
        normalized = re.sub(r'[&]', 'and', normalized)
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def get_journal_score(self, journal_name: str) -> float:
        """Get comprehensive journal score.
        
        Args:
            journal_name: Journal name to score
            
        Returns:
            float: Journal score (0-125 points)
        """
        if not journal_name:
            return 25.0  # Default minimum score
        
        normalized_name = self.normalize_journal_name(journal_name)
        
        # Try exact match first
        journal_info = self.journal_database.get(normalized_name)
        
        # If no exact match, try fuzzy matching
        if not journal_info:
            journal_info = self._fuzzy_match_journal(normalized_name)
        
        if not journal_info:
            logger.debug(f"Unknown journal: {journal_name}")
            return self._estimate_unknown_journal_score(journal_name)
        
        # Calculate base score
        score = journal_info.base_score
        
        # Add impact factor bonus
        if journal_info.impact_factor:
            if_bonus = min(math.log10(journal_info.impact_factor) * 10, 20.0)
            score += if_bonus
        
        # Add publisher prestige bonus
        if journal_info.publisher:
            prestige_bonus = self.publisher_prestige.get(journal_info.publisher, 0.0)
            score += prestige_bonus
        
        # Cap at maximum score
        score = min(score, 125.0)
        
        logger.debug(f"Journal '{journal_name}' scored {score:.1f} points")
        return score
    
    def get_journal_score_with_domain_bonus(self, journal_name: str, paper_content: str = "") -> float:
        """Get journal score with domain-specific bonus for pharmacogenomics.
        
        Args:
            journal_name: Journal name to score
            paper_content: Paper title/abstract for domain relevance detection
            
        Returns:
            float: Journal score with domain bonus applied
        """
        base_score = self.get_journal_score(journal_name)
        
        normalized_name = self.normalize_journal_name(journal_name)
        journal_info = self.journal_database.get(normalized_name)
        
        if not journal_info:
            journal_info = self._fuzzy_match_journal(normalized_name)
        
        # Apply domain relevance multiplier
        domain_multiplier = 1.0
        if journal_info:
            domain_multiplier = journal_info.domain_relevance
        
        # Additional bonus for pharmacogenomics content
        if paper_content:
            pgx_terms = [
                'pharmacogenomic', 'pharmacogenetic', 'cyp2d6', 'cyp2c19', 'cyp2c9',
                'warfarin', 'clopidogrel', 'precision medicine', 'personalized medicine',
                'drug metabolism', 'genetic variant', 'allele frequency'
            ]
            
            content_lower = paper_content.lower()
            pgx_matches = sum(1 for term in pgx_terms if term in content_lower)
            
            if pgx_matches >= 3:
                domain_multiplier *= 1.2
            elif pgx_matches >= 1:
                domain_multiplier *= 1.1
        
        final_score = base_score * domain_multiplier
        return min(final_score, 125.0)  # Cap at maximum
    
    def get_journal_tier(self, journal_name: str) -> int:
        """Get journal tier classification.
        
        Args:
            journal_name: Journal name
            
        Returns:
            int: Tier number (1-6, lower is better)
        """
        normalized_name = self.normalize_journal_name(journal_name)
        journal_info = self.journal_database.get(normalized_name)
        
        if not journal_info:
            journal_info = self._fuzzy_match_journal(normalized_name)
        
        if journal_info:
            return journal_info.tier.value
        
        return 6  # Default to lowest tier for unknown journals
    
    def get_impact_factor(self, journal_name: str) -> Optional[float]:
        """Get journal impact factor.
        
        Args:
            journal_name: Journal name
            
        Returns:
            Optional[float]: Impact factor if available
        """
        normalized_name = self.normalize_journal_name(journal_name)
        journal_info = self.journal_database.get(normalized_name)
        
        if not journal_info:
            journal_info = self._fuzzy_match_journal(normalized_name)
        
        return journal_info.impact_factor if journal_info else None
    
    def get_publisher_prestige(self, journal_name: str) -> float:
        """Get publisher prestige bonus.
        
        Args:
            journal_name: Journal name
            
        Returns:
            float: Publisher prestige bonus (0-5 points)
        """
        normalized_name = self.normalize_journal_name(journal_name)
        journal_info = self.journal_database.get(normalized_name)
        
        if not journal_info:
            journal_info = self._fuzzy_match_journal(normalized_name)
        
        if journal_info and journal_info.publisher:
            return self.publisher_prestige.get(journal_info.publisher, 0.0)
        
        return 0.0
    
    def _fuzzy_match_journal(self, normalized_name: str) -> Optional[JournalInfo]:
        """Attempt fuzzy matching for journal names.
        
        Args:
            normalized_name: Normalized journal name
            
        Returns:
            Optional[JournalInfo]: Best matching journal info
        """
        if not normalized_name:
            return None
        
        # Try partial matches
        for db_name, journal_info in self.journal_database.items():
            # Check if the input is a substring of a known journal
            if normalized_name in db_name or db_name in normalized_name:
                if len(normalized_name) > 3:  # Avoid very short matches
                    return journal_info
        
        # Try word-based matching
        input_words = set(normalized_name.split())
        best_match = None
        best_score = 0
        
        for db_name, journal_info in self.journal_database.items():
            db_words = set(db_name.split())
            
            # Calculate word overlap
            overlap = len(input_words.intersection(db_words))
            total_words = len(input_words.union(db_words))
            
            if total_words > 0:
                similarity = overlap / total_words
                if similarity > best_score and similarity > 0.5:
                    best_score = similarity
                    best_match = journal_info
        
        return best_match
    
    def _estimate_unknown_journal_score(self, journal_name: str) -> float:
        """Estimate score for unknown journals based on name patterns.
        
        Args:
            journal_name: Journal name
            
        Returns:
            float: Estimated score
        """
        name_lower = journal_name.lower()
        
        # High-quality indicators
        if any(term in name_lower for term in ['nature', 'science', 'cell', 'nejm', 'lancet']):
            return 85.0
        
        # Domain-specific indicators
        if any(term in name_lower for term in ['pharmacogenomic', 'pharmacogenetic', 'precision medicine']):
            return 65.0
        
        # Medical/clinical indicators
        if any(term in name_lower for term in ['clinical', 'medical', 'therapeutic']):
            return 55.0
        
        # Open access indicators (often lower impact)
        if any(term in name_lower for term in ['plos', 'bmc', 'frontiers', 'mdpi']):
            return 40.0
        
        # Default for unknown journals
        return 35.0
    
    def get_journal_statistics(self) -> Dict[str, any]:
        """Get statistics about the journal database.
        
        Returns:
            Dict[str, any]: Database statistics
        """
        tier_counts = {}
        impact_factors = []
        publishers = set()
        
        # Use a dict to remove duplicates by journal name
        unique_journals = {}
        for journal_info in self.journal_database.values():
            unique_journals[journal_info.name] = journal_info
        
        for journal_info in unique_journals.values():
            tier = journal_info.tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            if journal_info.impact_factor:
                impact_factors.append(journal_info.impact_factor)
            
            if journal_info.publisher:
                publishers.add(journal_info.publisher)
        
        return {
            'total_journals': len(unique_journals),
            'total_entries': len(self.journal_database),  # Including aliases
            'tier_distribution': tier_counts,
            'publishers': len(publishers),
            'journals_with_if': len(impact_factors),
            'avg_impact_factor': sum(impact_factors) / len(impact_factors) if impact_factors else 0,
            'max_impact_factor': max(impact_factors) if impact_factors else 0,
            'min_impact_factor': min(impact_factors) if impact_factors else 0
        }
    
    def search_journals(self, query: str, limit: int = 10) -> List[Tuple[str, float, int]]:
        """Search for journals matching a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List[Tuple[str, float, int]]: List of (journal_name, score, tier) tuples
        """
        query_normalized = self.normalize_journal_name(query)
        results = []
        
        # Get unique journals (avoid duplicates from aliases)
        unique_journals = {}
        for journal_info in self.journal_database.values():
            unique_journals[journal_info.name] = journal_info
        
        for journal_info in unique_journals.values():
            # Check if query matches journal name or aliases
            if (query_normalized in journal_info.normalized_name or 
                any(query_normalized in alias.lower() for alias in journal_info.aliases)):
                
                score = self.get_journal_score(journal_info.name)
                results.append((journal_info.name, score, journal_info.tier.value))
        
        # Sort by score (descending) and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]