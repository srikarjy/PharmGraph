"""Pharmacogenomics-specific feature extraction for ML pipeline."""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import json

from ..config import get_logger
from ..data_ingestion.data_models import PaperMetadata


logger = get_logger(__name__)


@dataclass
class DrugGeneInteraction:
    """Drug-gene interaction pair with metadata."""
    drug: str
    gene: str
    interaction_type: str  # 'substrate', 'inhibitor', 'inducer', 'transporter'
    confidence: float
    evidence_level: str  # 'strong', 'moderate', 'weak'
    clinical_significance: str  # 'high', 'medium', 'low'


@dataclass
class ClinicalTrialFeatures:
    """Clinical trial phase and regulatory features."""
    phase_mentions: Dict[str, int]
    regulatory_mentions: Dict[str, int]
    trial_design_features: Dict[str, int]
    endpoint_mentions: Dict[str, int]


@dataclass
class PopulationFeatures:
    """Population and ethnicity-related features."""
    ethnicity_mentions: Dict[str, int]
    population_size_mentions: List[int]
    geographic_mentions: Dict[str, int]
    age_group_mentions: Dict[str, int]
    gender_mentions: Dict[str, int]


class PharmacogenomicsPatterns:
    """Comprehensive patterns for pharmacogenomics entity recognition."""
    
    def __init__(self):
        """Initialize pharmacogenomics patterns."""
        
        # CYP enzyme patterns
        self.cyp_patterns = {
            'CYP1A1': r'\bCYP1A1\b',
            'CYP1A2': r'\bCYP1A2\b',
            'CYP2A6': r'\bCYP2A6\b',
            'CYP2B6': r'\bCYP2B6\b',
            'CYP2C8': r'\bCYP2C8\b',
            'CYP2C9': r'\bCYP2C9\b',
            'CYP2C19': r'\bCYP2C19\b',
            'CYP2D6': r'\bCYP2D6\b',
            'CYP2E1': r'\bCYP2E1\b',
            'CYP3A4': r'\bCYP3A4\b',
            'CYP3A5': r'\bCYP3A5\b'
        }
        
        # Drug patterns by therapeutic class
        self.drug_patterns = {
            # Cardiovascular drugs
            'warfarin': r'\b(warfarin|coumadin)\b',
            'clopidogrel': r'\b(clopidogrel|plavix)\b',
            'simvastatin': r'\b(simvastatin|zocor)\b',
            'atorvastatin': r'\b(atorvastatin|lipitor)\b',
            
            # Psychiatric drugs
            'sertraline': r'\b(sertraline|zoloft)\b',
            'fluoxetine': r'\b(fluoxetine|prozac)\b',
            'paroxetine': r'\b(paroxetine|paxil)\b',
            'venlafaxine': r'\b(venlafaxine|effexor)\b',
            
            # Pain medications
            'codeine': r'\bcodeine\b',
            'tramadol': r'\btramadol\b',
            'oxycodone': r'\boxycodone\b',
            'morphine': r'\bmorphine\b',
            
            # Proton pump inhibitors
            'omeprazole': r'\b(omeprazole|prilosec)\b',
            'esomeprazole': r'\b(esomeprazole|nexium)\b',
            'lansoprazole': r'\b(lansoprazole|prevacid)\b',
            
            # Oncology drugs
            'tamoxifen': r'\btamoxifen\b',
            'irinotecan': r'\birinotecan\b',
            'fluorouracil': r'\b(fluorouracil|5-FU|5-fluorouracil)\b',
            'capecitabine': r'\bcapecitabine\b',
            
            # Immunosuppressants
            'tacrolimus': r'\btacrolimus\b',
            'cyclosporine': r'\bcyclosporine\b',
            'azathioprine': r'\bazathioprine\b',
            
            # Antivirals
            'abacavir': r'\babacavir\b',
            'efavirenz': r'\befavirenz\b'
        }
        
        # Gene patterns beyond CYP enzymes
        self.gene_patterns = {
            'VKORC1': r'\bVKORC1\b',
            'DPYD': r'\bDPYD\b',
            'TPMT': r'\bTPMT\b',
            'UGT1A1': r'\bUGT1A1\b',
            'SLCO1B1': r'\bSLCO1B1\b',
            'ABCB1': r'\bABCB1\b',
            'COMT': r'\bCOMT\b',
            'HLA-B': r'\bHLA-B\*?\d+:\d+\b',
            'HLA-A': r'\bHLA-A\*?\d+:\d+\b',
            'CACNA1S': r'\bCACNA1S\b',
            'RYR1': r'\bRYR1\b'
        }
        
        # SNP patterns
        self.snp_patterns = {
            'rs_number': r'\brs\d+\b',
            'star_allele': r'\*\d+[A-Z]?',
            'nucleotide_change': r'\b[ATCG]\d+[ATCG]\b',
            'amino_acid_change': r'\b[A-Z]\d+[A-Z]\b'
        }
        
        # Clinical trial phase patterns
        self.trial_phase_patterns = {
            'phase_0': r'\bphase\s*0\b|\bphase\s*zero\b',
            'phase_1': r'\bphase\s*I\b|\bphase\s*1\b|\bphase\s*one\b',
            'phase_2': r'\bphase\s*II\b|\bphase\s*2\b|\bphase\s*two\b',
            'phase_3': r'\bphase\s*III\b|\bphase\s*3\b|\bphase\s*three\b',
            'phase_4': r'\bphase\s*IV\b|\bphase\s*4\b|\bphase\s*four\b'
        }
        
        # Regulatory agency patterns
        self.regulatory_patterns = {
            'FDA': r'\b(FDA|Food and Drug Administration)\b',
            'EMA': r'\b(EMA|European Medicines Agency)\b',
            'PMDA': r'\b(PMDA|Pharmaceuticals and Medical Devices Agency)\b',
            'Health_Canada': r'\bHealth Canada\b',
            'TGA': r'\b(TGA|Therapeutic Goods Administration)\b',
            'CPIC': r'\b(CPIC|Clinical Pharmacogenetics Implementation Consortium)\b',
            'PharmGKB': r'\bPharmGKB\b',
            'DPWG': r'\b(DPWG|Dutch Pharmacogenetics Working Group)\b'
        }
        
        # Population and ethnicity patterns
        self.population_patterns = {
            'caucasian': r'\b(caucasian|white|european)\b',
            'african': r'\b(african|black|afro)\b',
            'asian': r'\b(asian|chinese|japanese|korean|thai|vietnamese)\b',
            'hispanic': r'\b(hispanic|latino|latina)\b',
            'native_american': r'\b(native american|american indian)\b',
            'middle_eastern': r'\b(middle eastern|arab)\b',
            'south_asian': r'\b(south asian|indian|pakistani|bangladeshi)\b',
            'mixed': r'\b(mixed|multiracial|multi-ethnic)\b'
        }
        
        # Age group patterns
        self.age_patterns = {
            'pediatric': r'\b(pediatric|paediatric|children|child|infant|neonatal)\b',
            'adolescent': r'\b(adolescent|teenager|teen)\b',
            'adult': r'\b(adult|adults)\b',
            'elderly': r'\b(elderly|geriatric|senior)\b'
        }
        
        # Gender patterns
        self.gender_patterns = {
            'male': r'\b(male|men|man)\b',
            'female': r'\b(female|women|woman)\b'
        }


class DrugGeneInteractionExtractor:
    """Extracts drug-gene interaction pairs and relationships."""
    
    def __init__(self):
        """Initialize drug-gene interaction extractor."""
        self.patterns = PharmacogenomicsPatterns()
        
        # Known drug-gene interactions with metadata
        self.known_interactions = {
            ('warfarin', 'CYP2C9'): {
                'type': 'substrate',
                'evidence': 'strong',
                'significance': 'high'
            },
            ('warfarin', 'VKORC1'): {
                'type': 'target',
                'evidence': 'strong',
                'significance': 'high'
            },
            ('clopidogrel', 'CYP2C19'): {
                'type': 'substrate',
                'evidence': 'strong',
                'significance': 'high'
            },
            ('codeine', 'CYP2D6'): {
                'type': 'substrate',
                'evidence': 'strong',
                'significance': 'high'
            },
            ('tamoxifen', 'CYP2D6'): {
                'type': 'substrate',
                'evidence': 'strong',
                'significance': 'high'
            },
            ('simvastatin', 'SLCO1B1'): {
                'type': 'transporter',
                'evidence': 'strong',
                'significance': 'medium'
            }
        }
    
    def extract_drug_gene_pairs(self, text: str) -> List[DrugGeneInteraction]:
        """Extract drug-gene interaction pairs from text.
        
        Args:
            text: Input text
            
        Returns:
            List of drug-gene interactions
        """
        interactions = []
        text_lower = text.lower()
        
        # Find all drug mentions
        drug_mentions = {}
        for drug, pattern in self.patterns.drug_patterns.items():
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                drug_mentions[match.span()] = drug
        
        # Find all gene mentions
        gene_mentions = {}
        for gene, pattern in self.patterns.gene_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                gene_mentions[match.span()] = gene
        
        # Find CYP enzyme mentions
        for cyp, pattern in self.patterns.cyp_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                gene_mentions[match.span()] = cyp
        
        # Create interactions for co-occurring drugs and genes
        for drug_span, drug in drug_mentions.items():
            for gene_span, gene in gene_mentions.items():
                # Check if drug and gene are mentioned close to each other
                distance = min(
                    abs(drug_span[0] - gene_span[1]),
                    abs(gene_span[0] - drug_span[1])
                )
                
                if distance < 200:  # Within 200 characters
                    # Get interaction metadata if known
                    interaction_key = (drug, gene)
                    metadata = self.known_interactions.get(interaction_key, {
                        'type': 'unknown',
                        'evidence': 'weak',
                        'significance': 'low'
                    })
                    
                    # Calculate confidence based on distance and known interactions
                    confidence = 1.0 if interaction_key in self.known_interactions else 0.5
                    confidence *= max(0.1, 1.0 - distance / 200)
                    
                    interactions.append(DrugGeneInteraction(
                        drug=drug,
                        gene=gene,
                        interaction_type=metadata['type'],
                        confidence=confidence,
                        evidence_level=metadata['evidence'],
                        clinical_significance=metadata['significance']
                    ))
        
        return interactions
    
    def extract_cyp_substrate_relationships(self, text: str) -> Dict[str, List[str]]:
        """Extract CYP enzyme-substrate relationships.
        
        Args:
            text: Input text
            
        Returns:
            Dict mapping CYP enzymes to their substrates
        """
        relationships = defaultdict(list)
        text_lower = text.lower()
        
        # Look for substrate/inhibitor/inducer patterns
        substrate_patterns = [
            r'(\w+)\s+is\s+a\s+substrate\s+of\s+(CYP\w+)',
            r'(CYP\w+)\s+substrate\s+(\w+)',
            r'(\w+)\s+metabolized\s+by\s+(CYP\w+)',
            r'(CYP\w+)\s+metabolizes\s+(\w+)'
        ]
        
        for pattern in substrate_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                drug = match.group(1).lower()
                cyp = match.group(2).upper()
                
                # Validate that it's a known drug and CYP enzyme
                if any(drug in drug_pattern for drug_pattern in self.patterns.drug_patterns.values()):
                    if cyp in self.patterns.cyp_patterns:
                        relationships[cyp].append(drug)
        
        return dict(relationships)


class ClinicalTrialFeatureExtractor:
    """Extracts clinical trial and regulatory features."""
    
    def __init__(self):
        """Initialize clinical trial feature extractor."""
        self.patterns = PharmacogenomicsPatterns()
        
        # Trial design patterns
        self.trial_design_patterns = {
            'randomized': r'\b(randomized|randomised|RCT)\b',
            'controlled': r'\bcontrolled\b',
            'double_blind': r'\b(double.blind|double.blinded)\b',
            'single_blind': r'\b(single.blind|single.blinded)\b',
            'open_label': r'\b(open.label|open label)\b',
            'crossover': r'\bcrossover\b',
            'parallel': r'\bparallel\b',
            'multicenter': r'\b(multicenter|multicentre)\b',
            'prospective': r'\bprospective\b',
            'retrospective': r'\bretrospective\b'
        }
        
        # Clinical endpoint patterns
        self.endpoint_patterns = {
            'primary_endpoint': r'\bprimary\s+endpoint\b',
            'secondary_endpoint': r'\bsecondary\s+endpoint\b',
            'efficacy': r'\befficacy\b',
            'safety': r'\bsafety\b',
            'adverse_events': r'\b(adverse\s+events?|AE|SAE)\b',
            'mortality': r'\bmortality\b',
            'morbidity': r'\bmorbidity\b',
            'quality_of_life': r'\bquality\s+of\s+life\b',
            'biomarker': r'\bbiomarker\b',
            'pharmacokinetics': r'\b(pharmacokinetics|PK)\b',
            'pharmacodynamics': r'\b(pharmacodynamics|PD)\b'
        }
    
    def extract_clinical_trial_features(self, text: str) -> ClinicalTrialFeatures:
        """Extract clinical trial features from text.
        
        Args:
            text: Input text
            
        Returns:
            ClinicalTrialFeatures object
        """
        # Extract phase mentions
        phase_mentions = {}
        for phase, pattern in self.patterns.trial_phase_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            phase_mentions[phase] = matches
        
        # Extract regulatory mentions
        regulatory_mentions = {}
        for agency, pattern in self.patterns.regulatory_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            regulatory_mentions[agency] = matches
        
        # Extract trial design features
        trial_design_features = {}
        for design, pattern in self.trial_design_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            trial_design_features[design] = matches
        
        # Extract endpoint mentions
        endpoint_mentions = {}
        for endpoint, pattern in self.endpoint_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            endpoint_mentions[endpoint] = matches
        
        return ClinicalTrialFeatures(
            phase_mentions=phase_mentions,
            regulatory_mentions=regulatory_mentions,
            trial_design_features=trial_design_features,
            endpoint_mentions=endpoint_mentions
        )


class PopulationFeatureExtractor:
    """Extracts population and ethnicity-related features."""
    
    def __init__(self):
        """Initialize population feature extractor."""
        self.patterns = PharmacogenomicsPatterns()
        
        # Sample size patterns
        self.sample_size_patterns = [
            r'n\s*=\s*(\d+)',
            r'(\d+)\s+patients?',
            r'(\d+)\s+subjects?',
            r'(\d+)\s+participants?',
            r'sample\s+size\s+of\s+(\d+)',
            r'cohort\s+of\s+(\d+)'
        ]
        
        # Geographic patterns
        self.geographic_patterns = {
            'north_america': r'\b(USA|United States|Canada|North America)\b',
            'europe': r'\b(Europe|European|UK|Germany|France|Italy|Spain|Netherlands)\b',
            'asia': r'\b(Asia|Asian|China|Japan|Korea|India|Thailand|Singapore)\b',
            'africa': r'\b(Africa|African|Nigeria|South Africa|Kenya)\b',
            'south_america': r'\b(South America|Brazil|Argentina|Chile)\b',
            'oceania': r'\b(Australia|New Zealand|Oceania)\b'
        }
    
    def extract_population_features(self, text: str) -> PopulationFeatures:
        """Extract population features from text.
        
        Args:
            text: Input text
            
        Returns:
            PopulationFeatures object
        """
        # Extract ethnicity mentions
        ethnicity_mentions = {}
        for ethnicity, pattern in self.patterns.population_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            ethnicity_mentions[ethnicity] = matches
        
        # Extract population sizes
        population_sizes = []
        for pattern in self.sample_size_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    size = int(match)
                    if 10 <= size <= 100000:  # Reasonable range for clinical studies
                        population_sizes.append(size)
                except ValueError:
                    continue
        
        # Extract geographic mentions
        geographic_mentions = {}
        for region, pattern in self.geographic_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            geographic_mentions[region] = matches
        
        # Extract age group mentions
        age_mentions = {}
        for age_group, pattern in self.patterns.age_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            age_mentions[age_group] = matches
        
        # Extract gender mentions
        gender_mentions = {}
        for gender, pattern in self.patterns.gender_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            gender_mentions[gender] = matches
        
        return PopulationFeatures(
            ethnicity_mentions=ethnicity_mentions,
            population_size_mentions=population_sizes,
            geographic_mentions=geographic_mentions,
            age_group_mentions=age_mentions,
            gender_mentions=gender_mentions
        )


class PharmacogenomicsFeatureExtractor:
    """Main pharmacogenomics-specific feature extractor."""
    
    def __init__(self):
        """Initialize pharmacogenomics feature extractor."""
        self.patterns = PharmacogenomicsPatterns()
        self.drug_gene_extractor = DrugGeneInteractionExtractor()
        self.clinical_extractor = ClinicalTrialFeatureExtractor()
        self.population_extractor = PopulationFeatureExtractor()
        
        logger.info("PharmacogenomicsFeatureExtractor initialized")
    
    def extract_features(self, papers: List[PaperMetadata]) -> Dict[str, np.ndarray]:
        """Extract comprehensive pharmacogenomics features.
        
        Args:
            papers: List of paper metadata
            
        Returns:
            Dict of feature arrays
        """
        logger.info(f"Extracting pharmacogenomics features from {len(papers)} papers")
        
        features = defaultdict(list)
        
        for paper in papers:
            # Combine title and abstract
            text = ""
            if paper.title:
                text += paper.title + " "
            if paper.abstract:
                text += paper.abstract
            
            # Extract drug-gene interactions
            interactions = self.drug_gene_extractor.extract_drug_gene_pairs(text)
            
            # Count interactions by type
            interaction_counts = Counter([interaction.interaction_type for interaction in interactions])
            features['substrate_interactions'].append(interaction_counts.get('substrate', 0))
            features['inhibitor_interactions'].append(interaction_counts.get('inhibitor', 0))
            features['inducer_interactions'].append(interaction_counts.get('inducer', 0))
            features['transporter_interactions'].append(interaction_counts.get('transporter', 0))
            features['total_interactions'].append(len(interactions))
            
            # Average interaction confidence
            if interactions:
                avg_confidence = np.mean([interaction.confidence for interaction in interactions])
                features['avg_interaction_confidence'].append(avg_confidence)
            else:
                features['avg_interaction_confidence'].append(0.0)
            
            # Extract CYP enzyme mentions
            cyp_counts = {}
            for cyp, pattern in self.patterns.cyp_patterns.items():
                count = len(re.findall(pattern, text, re.IGNORECASE))
                cyp_counts[cyp] = count
                features[f'{cyp}_mentions'].append(count)
            
            features['total_cyp_mentions'].append(sum(cyp_counts.values()))
            
            # Extract drug mentions by class
            drug_class_counts = defaultdict(int)
            for drug, pattern in self.patterns.drug_patterns.items():
                count = len(re.findall(pattern, text, re.IGNORECASE))
                features[f'{drug}_mentions'].append(count)
                
                # Classify drugs by therapeutic area
                if drug in ['warfarin', 'clopidogrel', 'simvastatin', 'atorvastatin']:
                    drug_class_counts['cardiovascular'] += count
                elif drug in ['sertraline', 'fluoxetine', 'paroxetine', 'venlafaxine']:
                    drug_class_counts['psychiatric'] += count
                elif drug in ['codeine', 'tramadol', 'oxycodone', 'morphine']:
                    drug_class_counts['analgesic'] += count
                elif drug in ['tamoxifen', 'irinotecan', 'fluorouracil', 'capecitabine']:
                    drug_class_counts['oncology'] += count
            
            for drug_class, count in drug_class_counts.items():
                features[f'{drug_class}_drug_mentions'].append(count)
            
            # Extract gene mentions
            gene_counts = {}
            for gene, pattern in self.patterns.gene_patterns.items():
                count = len(re.findall(pattern, text, re.IGNORECASE))
                gene_counts[gene] = count
                features[f'{gene}_mentions'].append(count)
            
            features['total_gene_mentions'].append(sum(gene_counts.values()))
            
            # Extract SNP mentions
            snp_counts = {}
            for snp_type, pattern in self.patterns.snp_patterns.items():
                count = len(re.findall(pattern, text, re.IGNORECASE))
                snp_counts[snp_type] = count
                features[f'{snp_type}_mentions'].append(count)
            
            features['total_snp_mentions'].append(sum(snp_counts.values()))
            
            # Extract clinical trial features
            clinical_features = self.clinical_extractor.extract_clinical_trial_features(text)
            
            # Phase mentions
            for phase, count in clinical_features.phase_mentions.items():
                features[f'{phase}_mentions'].append(count)
            
            features['total_phase_mentions'].append(sum(clinical_features.phase_mentions.values()))
            
            # Regulatory mentions
            for agency, count in clinical_features.regulatory_mentions.items():
                features[f'{agency}_mentions'].append(count)
            
            features['total_regulatory_mentions'].append(sum(clinical_features.regulatory_mentions.values()))
            
            # Trial design features
            for design, count in clinical_features.trial_design_features.items():
                features[f'{design}_mentions'].append(count)
            
            # Endpoint mentions
            for endpoint, count in clinical_features.endpoint_mentions.items():
                features[f'{endpoint}_mentions'].append(count)
            
            # Extract population features
            population_features = self.population_extractor.extract_population_features(text)
            
            # Ethnicity mentions
            for ethnicity, count in population_features.ethnicity_mentions.items():
                features[f'{ethnicity}_mentions'].append(count)
            
            features['total_ethnicity_mentions'].append(sum(population_features.ethnicity_mentions.values()))
            
            # Population size features
            if population_features.population_size_mentions:
                features['max_population_size'].append(max(population_features.population_size_mentions))
                features['avg_population_size'].append(np.mean(population_features.population_size_mentions))
                features['population_size_count'].append(len(population_features.population_size_mentions))
            else:
                features['max_population_size'].append(0)
                features['avg_population_size'].append(0)
                features['population_size_count'].append(0)
            
            # Geographic mentions
            for region, count in population_features.geographic_mentions.items():
                features[f'{region}_mentions'].append(count)
            
            # Age and gender mentions
            for age_group, count in population_features.age_group_mentions.items():
                features[f'{age_group}_mentions'].append(count)
            
            for gender, count in population_features.gender_mentions.items():
                features[f'{gender}_mentions'].append(count)
            
            # Derived features
            features['drug_gene_ratio'].append(
                sum(drug_class_counts.values()) / max(1, sum(gene_counts.values()))
            )
            
            features['clinical_focus_score'].append(
                sum(clinical_features.phase_mentions.values()) + 
                sum(clinical_features.endpoint_mentions.values())
            )
            
            features['regulatory_focus_score'].append(
                sum(clinical_features.regulatory_mentions.values())
            )
            
            features['population_diversity_score'].append(
                len([count for count in population_features.ethnicity_mentions.values() if count > 0])
            )
        
        # Convert to numpy arrays
        feature_arrays = {name: np.array(values) for name, values in features.items()}
        
        logger.info(f"Extracted {len(feature_arrays)} pharmacogenomics features")
        return feature_arrays
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of extracted features.
        
        Returns:
            Dict mapping feature names to descriptions
        """
        descriptions = {
            'total_interactions': 'Total number of drug-gene interactions mentioned',
            'substrate_interactions': 'Number of substrate interactions',
            'inhibitor_interactions': 'Number of inhibitor interactions',
            'avg_interaction_confidence': 'Average confidence of drug-gene interactions',
            'total_cyp_mentions': 'Total CYP enzyme mentions',
            'total_gene_mentions': 'Total pharmacogenomics gene mentions',
            'total_snp_mentions': 'Total SNP/variant mentions',
            'cardiovascular_drug_mentions': 'Cardiovascular drug mentions',
            'psychiatric_drug_mentions': 'Psychiatric drug mentions',
            'analgesic_drug_mentions': 'Pain medication mentions',
            'oncology_drug_mentions': 'Cancer drug mentions',
            'total_phase_mentions': 'Clinical trial phase mentions',
            'total_regulatory_mentions': 'Regulatory agency mentions',
            'total_ethnicity_mentions': 'Population/ethnicity mentions',
            'max_population_size': 'Largest study population mentioned',
            'drug_gene_ratio': 'Ratio of drug to gene mentions',
            'clinical_focus_score': 'Clinical trial focus score',
            'regulatory_focus_score': 'Regulatory focus score',
            'population_diversity_score': 'Population diversity score'
        }
        
        return descriptions