"""Pharmacogenomics query builder for domain-specific NCBI searches."""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from ..config import get_logger


logger = get_logger(__name__)


class QueryTemplate(Enum):
    """Pre-defined query templates for pharmacogenomics research."""
    PGX_ML_BASIC = "pharmacogenomics_ml_basic"
    CYP_AI = "cyp_artificial_intelligence"
    DRUG_DOSING = "drug_dosing_algorithm"
    PRECISION_MEDICINE = "precision_medicine_interactions"
    PHARMACOKINETICS_ML = "pharmacokinetics_machine_learning"
    CLINICAL_TRIALS_PGX = "clinical_trials_pharmacogenomics"
    BIOMARKERS_AI = "biomarkers_artificial_intelligence"
    ADVERSE_REACTIONS_ML = "adverse_reactions_machine_learning"


@dataclass
class QueryComponents:
    """Components for building complex pharmacogenomics queries."""
    drug_terms: List[str]
    gene_terms: List[str]
    ml_terms: List[str]
    clinical_terms: List[str]
    mesh_terms: List[str]
    field_tags: List[str]


class PharmacogenomicsQueryBuilder:
    """Builder for domain-specific pharmacogenomics search queries."""
    
    def __init__(self):
        """Initialize the query builder with predefined terms."""
        
        # Core pharmacogenomics terms
        self.pgx_core_terms = [
            "pharmacogenomics", "pharmacogenetics", "personalized medicine",
            "precision medicine", "individualized therapy", "drug metabolism"
        ]
        
        # Machine learning and AI terms
        self.ml_ai_terms = [
            "machine learning", "artificial intelligence", "deep learning",
            "neural network", "random forest", "support vector machine",
            "gradient boosting", "ensemble methods", "predictive modeling",
            "classification algorithm", "regression analysis", "feature selection"
        ]
        
        # Common pharmacogenes
        self.pharmacogenes = [
            "CYP2D6", "CYP2C19", "CYP2C9", "CYP3A4", "CYP3A5", "CYP1A2",
            "DPYD", "TPMT", "UGT1A1", "SLCO1B1", "ABCB1", "VKORC1",
            "CFTR", "G6PD", "HLA-B", "HLA-A", "NUDT15", "RYR1"
        ]
        
        # Drug categories and specific drugs
        self.drug_categories = {
            "anticoagulants": ["warfarin", "dabigatran", "rivaroxaban", "apixaban"],
            "antiplatelets": ["clopidogrel", "prasugrel", "ticagrelor"],
            "antidepressants": ["sertraline", "paroxetine", "fluoxetine", "venlafaxine"],
            "antipsychotics": ["haloperidol", "risperidone", "olanzapine", "quetiapine"],
            "chemotherapy": ["5-fluorouracil", "irinotecan", "tamoxifen", "trastuzumab"],
            "immunosuppressants": ["tacrolimus", "cyclosporine", "azathioprine"],
            "cardiovascular": ["metoprolol", "propranolol", "atorvastatin", "simvastatin"],
            "pain_management": ["codeine", "tramadol", "morphine", "oxycodone"]
        }
        
        # Clinical study types
        self.study_types = [
            "clinical trial", "randomized controlled trial", "meta-analysis",
            "systematic review", "cohort study", "case-control study",
            "genome-wide association study", "pharmacokinetic study"
        ]
        
        # High-impact journals for filtering
        self.high_impact_journals = [
            "Nature", "Science", "Cell", "New England Journal of Medicine",
            "Lancet", "JAMA", "Nature Genetics", "Nature Medicine",
            "Clinical Pharmacology & Therapeutics", "Pharmacogenomics",
            "Pharmacogenetics and Genomics", "Journal of Personalized Medicine"
        ]
        
        logger.info("Initialized PharmacogenomicsQueryBuilder")
    
    def build_pgx_ml_query(self,
                          drug_terms: Optional[List[str]] = None,
                          gene_terms: Optional[List[str]] = None,
                          ml_terms: Optional[List[str]] = None,
                          date_range: Optional[Tuple[str, str]] = None,
                          use_mesh: bool = True) -> str:
        """Build a pharmacogenomics + machine learning query.
        
        Args:
            drug_terms: Specific drugs to include
            gene_terms: Specific genes to include
            ml_terms: Specific ML terms to include
            date_range: Date range tuple (start, end) in YYYY/MM/DD format
            use_mesh: Whether to use MeSH terms
            
        Returns:
            str: Formatted search query
        """
        query_parts = []
        
        # Core pharmacogenomics terms
        if use_mesh:
            pgx_part = '("pharmacogenomics"[MeSH] OR "precision medicine"[MeSH])'
        else:
            pgx_part = '("pharmacogenomics"[Title/Abstract] OR "precision medicine"[Title/Abstract])'
        query_parts.append(pgx_part)
        
        # Machine learning terms
        if ml_terms:
            ml_selected = ml_terms
        else:
            ml_selected = ["machine learning", "artificial intelligence", "deep learning"]
        
        ml_part = " OR ".join([f'"{term}"[Title/Abstract]' for term in ml_selected])
        query_parts.append(f"({ml_part})")
        
        # Drug terms if specified
        if drug_terms:
            drug_part = " OR ".join([f'"{drug}"[All Fields]' for drug in drug_terms])
            query_parts.append(f"({drug_part})")
        
        # Gene terms if specified
        if gene_terms:
            gene_part = " OR ".join([f'"{gene}"[All Fields]' for gene in gene_terms])
            query_parts.append(f"({gene_part})")
        
        # Combine with AND
        base_query = " AND ".join(query_parts)
        
        # Add date range if specified
        if date_range:
            start_date, end_date = date_range
            date_filter = f' AND ("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
            base_query += date_filter
        
        logger.debug(f"Built PGx-ML query: {base_query}")
        return base_query
    
    def build_cyp_ai_query(self,
                          cyp_enzymes: Optional[List[str]] = None,
                          ai_terms: Optional[List[str]] = None,
                          include_substrates: bool = True) -> str:
        """Build a CYP enzyme + AI query.
        
        Args:
            cyp_enzymes: Specific CYP enzymes to include
            ai_terms: Specific AI terms to include
            include_substrates: Whether to include substrate terms
            
        Returns:
            str: Formatted search query
        """
        if cyp_enzymes is None:
            cyp_enzymes = ["CYP2D6", "CYP2C19", "CYP2C9", "CYP3A4"]
        
        if ai_terms is None:
            ai_terms = ["artificial intelligence", "deep learning", "neural network"]
        
        # CYP enzyme terms
        cyp_part = " OR ".join([f'"{cyp}"[All Fields]' for cyp in cyp_enzymes])
        
        # AI terms
        ai_part = " OR ".join([f'"{term}"[Title/Abstract]' for term in ai_terms])
        
        query = f"({cyp_part}) AND ({ai_part})"
        
        # Add substrate terms if requested
        if include_substrates:
            substrate_terms = ["substrate", "metabolism", "metabolizer", "phenotype"]
            substrate_part = " OR ".join([f'"{term}"[All Fields]' for term in substrate_terms])
            query += f" AND ({substrate_part})"
        
        logger.debug(f"Built CYP-AI query: {query}")
        return query
    
    def build_drug_dosing_query(self,
                               drugs: Optional[List[str]] = None,
                               algorithm_types: Optional[List[str]] = None) -> str:
        """Build a drug dosing algorithm query.
        
        Args:
            drugs: Specific drugs to include
            algorithm_types: Types of algorithms to include
            
        Returns:
            str: Formatted search query
        """
        if drugs is None:
            drugs = ["warfarin", "clopidogrel", "tacrolimus", "5-fluorouracil"]
        
        if algorithm_types is None:
            algorithm_types = ["dosing algorithm", "dose prediction", "dose optimization"]
        
        # Drug terms
        drug_part = " OR ".join([f'"{drug}"[All Fields]' for drug in drugs])
        
        # Algorithm terms
        algo_part = " OR ".join([f'"{algo}"[All Fields]' for algo in algorithm_types])
        
        query = f"({drug_part}) AND ({algo_part})"
        
        logger.debug(f"Built drug dosing query: {query}")
        return query
    
    def build_precision_medicine_query(self,
                                     focus_area: str = "drug_interactions",
                                     include_biomarkers: bool = True) -> str:
        """Build a precision medicine query.
        
        Args:
            focus_area: Focus area (drug_interactions, adverse_events, efficacy)
            include_biomarkers: Whether to include biomarker terms
            
        Returns:
            str: Formatted search query
        """
        base_query = '"precision medicine"[MeSH]'
        
        if focus_area == "drug_interactions":
            focus_terms = ["drug interaction", "drug-drug interaction", "polypharmacy"]
        elif focus_area == "adverse_events":
            focus_terms = ["adverse drug reaction", "side effect", "toxicity"]
        elif focus_area == "efficacy":
            focus_terms = ["drug efficacy", "treatment response", "therapeutic outcome"]
        else:
            focus_terms = ["drug interaction"]
        
        focus_part = " OR ".join([f'"{term}"[All Fields]' for term in focus_terms])
        query = f"{base_query} AND ({focus_part})"
        
        if include_biomarkers:
            biomarker_terms = ["biomarker", "genetic variant", "polymorphism", "genotype"]
            biomarker_part = " OR ".join([f'"{term}"[All Fields]' for term in biomarker_terms])
            query += f" AND ({biomarker_part})"
        
        logger.debug(f"Built precision medicine query: {query}")
        return query
    
    def build_pharmacokinetics_ml_query(self,
                                       pk_parameters: Optional[List[str]] = None,
                                       ml_methods: Optional[List[str]] = None) -> str:
        """Build a pharmacokinetics + machine learning query.
        
        Args:
            pk_parameters: PK parameters to include
            ml_methods: ML methods to include
            
        Returns:
            str: Formatted search query
        """
        if pk_parameters is None:
            pk_parameters = ["clearance", "half-life", "bioavailability", "Cmax", "AUC"]
        
        if ml_methods is None:
            ml_methods = ["neural network", "random forest", "support vector machine"]
        
        # Base pharmacokinetics term
        pk_base = '"pharmacokinetics"[MeSH]'
        
        # PK parameters
        pk_param_part = " OR ".join([f'"{param}"[All Fields]' for param in pk_parameters])
        
        # ML methods
        ml_part = " OR ".join([f'"{method}"[Title/Abstract]' for method in ml_methods])
        
        query = f"{pk_base} AND ({pk_param_part}) AND ({ml_part})"
        
        logger.debug(f"Built PK-ML query: {query}")
        return query
    
    def build_clinical_trial_pgx_query(self,
                                      phase: Optional[str] = None,
                                      intervention_type: str = "drug") -> str:
        """Build a clinical trial pharmacogenomics query.
        
        Args:
            phase: Clinical trial phase (I, II, III, IV)
            intervention_type: Type of intervention
            
        Returns:
            str: Formatted search query
        """
        base_query = '"clinical trial"[Publication Type] AND "pharmacogenomics"[MeSH]'
        
        if phase:
            phase_term = f'"phase {phase}"[All Fields]'
            base_query += f" AND {phase_term}"
        
        if intervention_type == "drug":
            intervention_terms = ["drug therapy", "pharmaceutical", "medication"]
            intervention_part = " OR ".join([f'"{term}"[All Fields]' for term in intervention_terms])
            base_query += f" AND ({intervention_part})"
        
        logger.debug(f"Built clinical trial PGx query: {base_query}")
        return base_query
    
    def build_adverse_reactions_ml_query(self,
                                        reaction_types: Optional[List[str]] = None,
                                        prediction_focus: bool = True) -> str:
        """Build an adverse drug reactions + ML query.
        
        Args:
            reaction_types: Types of adverse reactions
            prediction_focus: Whether to focus on prediction
            
        Returns:
            str: Formatted search query
        """
        if reaction_types is None:
            reaction_types = ["hepatotoxicity", "cardiotoxicity", "nephrotoxicity", "skin reaction"]
        
        # Base ADR terms
        adr_base = '"adverse drug reaction"[MeSH] OR "drug toxicity"[MeSH]'
        
        # Specific reaction types
        reaction_part = " OR ".join([f'"{reaction}"[All Fields]' for reaction in reaction_types])
        
        # ML terms for prediction
        ml_terms = ["machine learning", "predictive model", "risk prediction", "early warning"]
        ml_part = " OR ".join([f'"{term}"[Title/Abstract]' for term in ml_terms])
        
        query = f"({adr_base}) AND ({reaction_part}) AND ({ml_part})"
        
        if prediction_focus:
            prediction_terms = ["prediction", "risk assessment", "early detection"]
            pred_part = " OR ".join([f'"{term}"[All Fields]' for term in prediction_terms])
            query += f" AND ({pred_part})"
        
        logger.debug(f"Built ADR-ML query: {query}")
        return query
    
    def build_template_query(self, 
                           template: QueryTemplate,
                           custom_params: Optional[Dict[str, Any]] = None) -> str:
        """Build a query from a predefined template.
        
        Args:
            template: Query template to use
            custom_params: Custom parameters for the template
            
        Returns:
            str: Formatted search query
        """
        params = custom_params or {}
        
        if template == QueryTemplate.PGX_ML_BASIC:
            return self.build_pgx_ml_query(**params)
        elif template == QueryTemplate.CYP_AI:
            return self.build_cyp_ai_query(**params)
        elif template == QueryTemplate.DRUG_DOSING:
            return self.build_drug_dosing_query(**params)
        elif template == QueryTemplate.PRECISION_MEDICINE:
            return self.build_precision_medicine_query(**params)
        elif template == QueryTemplate.PHARMACOKINETICS_ML:
            return self.build_pharmacokinetics_ml_query(**params)
        elif template == QueryTemplate.CLINICAL_TRIALS_PGX:
            return self.build_clinical_trial_pgx_query(**params)
        elif template == QueryTemplate.ADVERSE_REACTIONS_ML:
            return self.build_adverse_reactions_ml_query(**params)
        else:
            logger.error(f"Unknown template: {template}")
            return ""
    
    def get_high_impact_journal_filter(self, journals: Optional[List[str]] = None) -> str:
        """Get a journal filter for high-impact publications.
        
        Args:
            journals: Specific journals to include (uses default if None)
            
        Returns:
            str: Journal filter string
        """
        if journals is None:
            journals = self.high_impact_journals[:10]  # Top 10 journals
        
        journal_part = " OR ".join([f'"{journal}"[Journal]' for journal in journals])
        return f"({journal_part})"
    
    def get_recent_years_filter(self, years_back: int = 5) -> str:
        """Get a filter for recent years.
        
        Args:
            years_back: Number of years back from current year
            
        Returns:
            str: Date filter string
        """
        from datetime import datetime, timedelta
        
        current_year = datetime.now().year
        start_year = current_year - years_back
        
        return f'("{start_year}"[Date - Publication] : "{current_year}"[Date - Publication])'
    
    def combine_queries(self, 
                       queries: List[str], 
                       operator: str = "OR") -> str:
        """Combine multiple queries with specified operator.
        
        Args:
            queries: List of query strings
            operator: Operator to use (AND, OR, NOT)
            
        Returns:
            str: Combined query string
        """
        if not queries:
            return ""
        
        if len(queries) == 1:
            return queries[0]
        
        # Wrap each query in parentheses and combine
        wrapped_queries = [f"({query})" for query in queries]
        combined = f" {operator} ".join(wrapped_queries)
        
        logger.debug(f"Combined {len(queries)} queries with {operator}")
        return combined
    
    def get_available_templates(self) -> List[str]:
        """Get list of available query templates.
        
        Returns:
            List[str]: List of template names
        """
        return [template.value for template in QueryTemplate]
    
    def get_pharmacogenes_by_category(self, category: str) -> List[str]:
        """Get pharmacogenes by functional category.
        
        Args:
            category: Gene category (cyp, transporter, target, etc.)
            
        Returns:
            List[str]: List of genes in the category
        """
        gene_categories = {
            "cyp": ["CYP2D6", "CYP2C19", "CYP2C9", "CYP3A4", "CYP3A5", "CYP1A2"],
            "transporter": ["SLCO1B1", "ABCB1", "ABCC2", "ABCG2"],
            "target": ["VKORC1", "DPYD", "TPMT", "UGT1A1"],
            "hla": ["HLA-B", "HLA-A", "HLA-DRB1", "HLA-DQA1"],
            "other": ["G6PD", "CFTR", "RYR1", "NUDT15"]
        }
        
        return gene_categories.get(category.lower(), [])