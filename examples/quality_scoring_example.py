"""Example demonstrating the complete quality scoring system for pharmacogenomics papers."""

import asyncio
from pathlib import Path
import sys
from datetime import datetime
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quality.quality_scorer import PharmacogenomicsQualityScorer, QualityScoreComponents
from src.quality.journal_scorer import JournalImpactScorer
from src.data_ingestion.data_models import PaperMetadata, Author, MeshTerm, Journal
from src.config import get_logger


logger = get_logger(__name__)


def create_sample_papers() -> List[PaperMetadata]:
    """Create sample papers for quality scoring demonstration."""
    
    papers = []
    
    # Paper 1: High-quality pharmacogenomics + ML paper
    paper1 = PaperMetadata(
        pmid='12345',
        title='Deep Learning for CYP2D6 Pharmacogenomics-Guided Warfarin Dosing: A Randomized Clinical Trial',
        abstract='''Background: Warfarin dosing remains challenging due to genetic variability in CYP2D6 metabolism. 
                   Methods: We developed a deep neural network using machine learning to predict optimal warfarin doses 
                   based on CYP2D6 genotypes. Cross-validation and feature selection were used to optimize model performance.
                   Results: Our artificial intelligence approach achieved 85% accuracy in dose prediction with external validation.
                   Conclusions: Machine learning can improve precision medicine approaches to warfarin dosing.''',
        authors=[
            Author(last_name='Johnson', first_name='Sarah', affiliation='Harvard Medical School'),
            Author(last_name='Chen', first_name='Wei', affiliation='MIT Computer Science'),
            Author(last_name='Rodriguez', first_name='Maria', affiliation='Mayo Clinic')
        ],
        journal=Journal(title='Clinical Pharmacology & Therapeutics', volume='115', issue='3'),
        publication_date=datetime(2024, 3, 15),
        mesh_terms=[
            MeshTerm(descriptor_name='Pharmacogenetics', major_topic=True),
            MeshTerm(descriptor_name='Machine Learning', major_topic=True),
            MeshTerm(descriptor_name='Warfarin', major_topic=True),
            MeshTerm(descriptor_name='Cytochrome P-450 CYP2D6', major_topic=True),
            MeshTerm(descriptor_name='Precision Medicine', major_topic=False)
        ],
        keywords=['pharmacogenomics', 'deep learning', 'warfarin', 'CYP2D6', 'precision medicine'],
        publication_types=['Journal Article', 'Randomized Controlled Trial'],
        doi='10.1002/cpt.2024.12345'
    )
    papers.append(paper1)
    
    # Paper 2: Nature Genetics paper (high journal impact)
    paper2 = PaperMetadata(
        pmid='67890',
        title='Genome-wide association study identifies novel pharmacogenomic variants affecting drug metabolism',
        abstract='''We conducted a genome-wide association study (GWAS) of 50,000 individuals to identify genetic variants 
                   affecting drug metabolism. Novel variants in CYP2C19 and DPYD were associated with adverse drug reactions.
                   These findings have implications for personalized medicine and clinical implementation.''',
        authors=[
            Author(last_name='Smith', first_name='John', affiliation='Broad Institute'),
            Author(last_name='Williams', first_name='Emma', affiliation='Wellcome Sanger Institute')
        ],
        journal=Journal(title='Nature Genetics', volume='56', issue='4'),
        publication_date=datetime(2024, 4, 1),
        mesh_terms=[
            MeshTerm(descriptor_name='Genome-Wide Association Study', major_topic=True),
            MeshTerm(descriptor_name='Pharmacogenetics', major_topic=True),
            MeshTerm(descriptor_name='Polymorphism, Single Nucleotide', major_topic=False)
        ],
        keywords=['GWAS', 'pharmacogenomics', 'CYP2C19', 'DPYD'],
        publication_types=['Journal Article'],
        doi='10.1038/ng.2024.67890'
    )
    papers.append(paper2)
    
    # Paper 3: Medium-quality bioinformatics paper
    paper3 = PaperMetadata(
        pmid='11111',
        title='Bioinformatics analysis of pharmacogenomic variants using random forest algorithms',
        abstract='''We applied random forest machine learning algorithms to analyze pharmacogenomic variants from public databases.
                   Feature importance analysis revealed key genetic markers for drug response prediction.''',
        authors=[
            Author(last_name='Zhang', first_name='Li', affiliation='University of California')
        ],
        journal=Journal(title='Bioinformatics', volume='40', issue='8'),
        publication_date=datetime(2023, 6, 1),
        mesh_terms=[
            MeshTerm(descriptor_name='Computational Biology', major_topic=True),
            MeshTerm(descriptor_name='Algorithms', major_topic=False)
        ],
        keywords=['bioinformatics', 'random forest', 'pharmacogenomics'],
        publication_types=['Journal Article']
    )
    papers.append(paper3)
    
    # Paper 4: Lower-quality open access paper
    paper4 = PaperMetadata(
        pmid='22222',
        title='A study of genetic variants in a small population',
        abstract='''We studied genetic variants in 100 patients and found some associations with drug response.
                   More research is needed to validate these findings.''',
        authors=[
            Author(last_name='Brown', first_name='David')
        ],
        journal=Journal(title='PLoS ONE', volume='18', issue='12'),
        publication_date=datetime(2023, 12, 1),
        mesh_terms=[
            MeshTerm(descriptor_name='Genetic Variation', major_topic=True)
        ],
        keywords=['genetics', 'drug response'],
        publication_types=['Journal Article']
    )
    papers.append(paper4)
    
    # Paper 5: Older paper (lower recency score)
    paper5 = PaperMetadata(
        pmid='33333',
        title='Traditional pharmacogenetic analysis of CYP2D6 polymorphisms',
        abstract='''This study examined CYP2D6 polymorphisms using traditional genetic analysis methods.
                   We found associations between genotype and drug metabolism phenotype.''',
        authors=[
            Author(last_name='Wilson', first_name='Robert', affiliation='University of Medicine')
        ],
        journal=Journal(title='Pharmacogenetics and Genomics', volume='25', issue='2'),
        publication_date=datetime(2015, 2, 1),  # Older paper
        mesh_terms=[
            MeshTerm(descriptor_name='Pharmacogenetics', major_topic=True),
            MeshTerm(descriptor_name='Cytochrome P-450 CYP2D6', major_topic=True)
        ],
        keywords=['CYP2D6', 'polymorphisms', 'pharmacogenetics'],
        publication_types=['Journal Article']
    )
    papers.append(paper5)
    
    return papers


def demonstrate_journal_scoring():
    """Demonstrate journal impact scoring system."""
    print("=== Journal Impact Scoring Demonstration ===\n")
    
    scorer = JournalImpactScorer()
    
    # Test various journals
    test_journals = [
        "Nature",
        "Science", 
        "Clinical Pharmacology & Therapeutics",
        "Pharmacogenomics",
        "Bioinformatics",
        "PLoS ONE",
        "Unknown Journal"
    ]
    
    print("Journal Scoring Results:")
    print("-" * 60)
    print(f"{'Journal':<40} {'Score':<8} {'Tier':<6} {'IF':<8}")
    print("-" * 60)
    
    for journal in test_journals:
        score = scorer.get_journal_score(journal)
        tier = scorer.get_journal_tier(journal)
        impact_factor = scorer.get_impact_factor(journal)
        if_str = f"{impact_factor:.1f}" if impact_factor else "N/A"
        
        print(f"{journal:<40} {score:<8.1f} {tier:<6} {if_str:<8}")
    
    print()
    
    # Test domain bonus
    print("Domain Bonus Demonstration:")
    print("-" * 40)
    
    pgx_content = "pharmacogenomics CYP2D6 warfarin precision medicine"
    regular_content = "general medical research study"
    
    test_journal = "Pharmacogenomics"
    base_score = scorer.get_journal_score(test_journal)
    pgx_score = scorer.get_journal_score_with_domain_bonus(test_journal, pgx_content)
    regular_score = scorer.get_journal_score_with_domain_bonus(test_journal, regular_content)
    
    print(f"Journal: {test_journal}")
    print(f"Base score: {base_score:.1f}")
    print(f"With PGx content: {pgx_score:.1f} (+{pgx_score - base_score:.1f})")
    print(f"With regular content: {regular_score:.1f} (+{regular_score - base_score:.1f})")
    print()
    
    # Database statistics
    stats = scorer.get_journal_statistics()
    print("Journal Database Statistics:")
    print(f"  Total journals: {stats['total_journals']}")
    print(f"  Total entries (with aliases): {stats['total_entries']}")
    print(f"  Publishers: {stats['publishers']}")
    print(f"  Journals with impact factors: {stats['journals_with_if']}")
    print(f"  Average impact factor: {stats['avg_impact_factor']:.1f}")
    print(f"  Tier distribution: {stats['tier_distribution']}")
    print()


def demonstrate_quality_scoring():
    """Demonstrate comprehensive quality scoring."""
    print("=== Comprehensive Quality Scoring Demonstration ===\n")
    
    scorer = PharmacogenomicsQualityScorer()
    papers = create_sample_papers()
    
    print("Scoring Individual Papers:")
    print("=" * 80)
    
    paper_scores = []
    
    for i, paper in enumerate(papers, 1):
        print(f"\nPaper {i}: {paper.title[:60]}...")
        print(f"PMID: {paper.pmid}")
        print(f"Journal: {paper.journal.title if paper.journal else 'Unknown'}")
        print(f"Publication Date: {paper.publication_date.strftime('%Y-%m-%d') if paper.publication_date else 'Unknown'}")
        print(f"Authors: {len(paper.authors)}")
        print(f"MeSH Terms: {len(paper.mesh_terms)}")
        print(f"Keywords: {', '.join(paper.keywords[:5])}")
        
        # Score the paper
        scores = scorer.score_paper(paper)
        paper_scores.append((paper, scores))
        
        print(f"\nQuality Scores:")
        print(f"  Journal Score: {scores.journal_score:.1f}/25.0")
        print(f"  Clinical Relevance: {scores.clinical_relevance_score:.1f}/25.0")
        print(f"  ML Methodology: {scores.ml_methodology_score:.1f}/25.0")
        print(f"  Recency: {scores.recency_score:.1f}/25.0")
        print(f"  TOTAL SCORE: {scores.total_score:.1f}/100.0")
        print(f"  Confidence: {scores.confidence_level:.2f}")
        
        # Quality tier
        if scores.total_score >= 75:
            quality_tier = "HIGH QUALITY"
        elif scores.total_score >= 50:
            quality_tier = "MEDIUM QUALITY"
        else:
            quality_tier = "LOW QUALITY"
        
        print(f"  Quality Tier: {quality_tier}")
        print("-" * 80)
    
    # Batch scoring demonstration
    print("\n=== Batch Scoring Analysis ===\n")
    
    all_scores = [scores for _, scores in paper_scores]
    distribution = scorer.get_quality_distribution(all_scores)
    
    print("Quality Distribution Analysis:")
    print(f"  Total Papers: {distribution['total_papers']}")
    print(f"  High Quality (≥75): {distribution['quality_distribution']['high_quality']}")
    print(f"  Medium Quality (50-74): {distribution['quality_distribution']['medium_quality']}")
    print(f"  Low Quality (<50): {distribution['quality_distribution']['low_quality']}")
    print(f"  High Quality Percentage: {distribution['quality_distribution']['high_quality_percentage']:.1f}%")
    print(f"  Average Confidence: {distribution['average_confidence']:.2f}")
    
    print(f"\nScore Statistics:")
    total_stats = distribution['total_score_stats']
    print(f"  Mean Total Score: {total_stats['mean']:.1f}")
    print(f"  Min Total Score: {total_stats['min']:.1f}")
    print(f"  Max Total Score: {total_stats['max']:.1f}")
    print(f"  Median Total Score: {total_stats['median']:.1f}")
    
    print(f"\nComponent Score Averages:")
    comp_stats = distribution['component_stats']
    print(f"  Journal: {comp_stats['journal']['mean']:.1f}/25.0")
    print(f"  Clinical Relevance: {comp_stats['clinical_relevance']['mean']:.1f}/25.0")
    print(f"  ML Methodology: {comp_stats['ml_methodology']['mean']:.1f}/25.0")
    print(f"  Recency: {comp_stats['recency']['mean']:.1f}/25.0")
    
    # Top papers
    print(f"\n=== Top 3 Papers by Quality Score ===\n")
    
    top_papers = scorer.get_top_papers(paper_scores, limit=3)
    
    for i, (paper, scores) in enumerate(top_papers, 1):
        print(f"{i}. {paper.title[:70]}...")
        print(f"   PMID: {paper.pmid} | Score: {scores.total_score:.1f} | Journal: {paper.journal.title if paper.journal else 'Unknown'}")
        print(f"   Breakdown: J={scores.journal_score:.1f}, C={scores.clinical_relevance_score:.1f}, M={scores.ml_methodology_score:.1f}, R={scores.recency_score:.1f}")
        print()


def demonstrate_scoring_factors():
    """Demonstrate how different factors affect scoring."""
    print("=== Scoring Factor Analysis ===\n")
    
    scorer = PharmacogenomicsQualityScorer()
    
    # Base paper
    base_paper = PaperMetadata(
        pmid='base',
        title='Basic pharmacogenomics study',
        abstract='This study examines genetic variants and drug response.',
        journal=Journal(title='Test Journal'),
        publication_date=datetime(2023, 1, 1),
        authors=[Author(last_name='Test')]
    )
    
    base_scores = scorer.score_paper(base_paper)
    print(f"Base Paper Score: {base_scores.total_score:.1f}")
    print(f"  Journal: {base_scores.journal_score:.1f}")
    print(f"  Clinical: {base_scores.clinical_relevance_score:.1f}")
    print(f"  ML: {base_scores.ml_methodology_score:.1f}")
    print(f"  Recency: {base_scores.recency_score:.1f}")
    print()
    
    # Factor 1: High-impact journal
    journal_paper = base_paper
    journal_paper.journal = Journal(title='Nature')
    journal_scores = scorer.score_paper(journal_paper)
    
    print(f"With High-Impact Journal (Nature): {journal_scores.total_score:.1f} (+{journal_scores.total_score - base_scores.total_score:.1f})")
    print(f"  Journal improvement: +{journal_scores.journal_score - base_scores.journal_score:.1f}")
    print()
    
    # Factor 2: Rich pharmacogenomics content
    pgx_paper = base_paper
    pgx_paper.title = 'CYP2D6 pharmacogenomics and warfarin dosing in precision medicine'
    pgx_paper.abstract = '''This randomized controlled trial investigates CYP2D6 variants and warfarin dosing.
                           We studied pharmacogenomics biomarkers for personalized medicine approaches.'''
    pgx_paper.mesh_terms = [
        MeshTerm(descriptor_name='Pharmacogenetics', major_topic=True),
        MeshTerm(descriptor_name='Warfarin', major_topic=True)
    ]
    pgx_scores = scorer.score_paper(pgx_paper)
    
    print(f"With Rich PGx Content: {pgx_scores.total_score:.1f} (+{pgx_scores.total_score - base_scores.total_score:.1f})")
    print(f"  Clinical relevance improvement: +{pgx_scores.clinical_relevance_score - base_scores.clinical_relevance_score:.1f}")
    print()
    
    # Factor 3: Machine learning methodology
    ml_paper = base_paper
    ml_paper.title = 'Deep learning and neural networks for pharmacogenomics prediction'
    ml_paper.abstract = '''We developed machine learning algorithms using deep neural networks and random forest
                          with cross-validation and feature selection for drug response prediction.'''
    ml_scores = scorer.score_paper(ml_paper)
    
    print(f"With ML Methodology: {ml_scores.total_score:.1f} (+{ml_scores.total_score - base_scores.total_score:.1f})")
    print(f"  ML methodology improvement: +{ml_scores.ml_methodology_score - base_scores.ml_methodology_score:.1f}")
    print()
    
    # Factor 4: Recent publication
    recent_paper = base_paper
    recent_paper.publication_date = datetime(2024, 6, 1)
    recent_scores = scorer.score_paper(recent_paper)
    
    print(f"With Recent Publication (2024): {recent_scores.total_score:.1f} (+{recent_scores.total_score - base_scores.total_score:.1f})")
    print(f"  Recency improvement: +{recent_scores.recency_score - base_scores.recency_score:.1f}")
    print()
    
    # Combined factors
    combined_paper = PaperMetadata(
        pmid='combined',
        title='Deep learning for CYP2D6 pharmacogenomics-guided warfarin dosing: A randomized trial',
        abstract='''This randomized controlled trial uses deep neural networks and machine learning
                   to predict warfarin dosing based on CYP2D6 pharmacogenomics variants. Cross-validation
                   and feature selection optimize our artificial intelligence approach for precision medicine.''',
        journal=Journal(title='Clinical Pharmacology & Therapeutics'),
        publication_date=datetime(2024, 6, 1),
        authors=[Author(last_name='Expert', first_name='Dr')],
        mesh_terms=[
            MeshTerm(descriptor_name='Pharmacogenetics', major_topic=True),
            MeshTerm(descriptor_name='Machine Learning', major_topic=True),
            MeshTerm(descriptor_name='Warfarin', major_topic=True)
        ],
        keywords=['pharmacogenomics', 'deep learning', 'warfarin', 'CYP2D6'],
        publication_types=['Journal Article', 'Randomized Controlled Trial']
    )
    
    combined_scores = scorer.score_paper(combined_paper)
    
    print(f"Combined High-Quality Paper: {combined_scores.total_score:.1f} (+{combined_scores.total_score - base_scores.total_score:.1f})")
    print(f"  Journal: {combined_scores.journal_score:.1f} (+{combined_scores.journal_score - base_scores.journal_score:.1f})")
    print(f"  Clinical: {combined_scores.clinical_relevance_score:.1f} (+{combined_scores.clinical_relevance_score - base_scores.clinical_relevance_score:.1f})")
    print(f"  ML: {combined_scores.ml_methodology_score:.1f} (+{combined_scores.ml_methodology_score - base_scores.ml_methodology_score:.1f})")
    print(f"  Recency: {combined_scores.recency_score:.1f} (+{combined_scores.recency_score - base_scores.recency_score:.1f})")
    print(f"  Confidence: {combined_scores.confidence_level:.2f}")


def main():
    """Main demonstration function."""
    try:
        print("Pharmacogenomics Quality Scoring System Demo")
        print("=" * 50)
        print()
        
        demonstrate_journal_scoring()
        demonstrate_quality_scoring()
        demonstrate_scoring_factors()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Comprehensive journal impact scoring with 80+ journals")
        print("✅ Domain-specific bonuses for pharmacogenomics content")
        print("✅ Multi-factor quality scoring (journal, clinical, ML, recency)")
        print("✅ Confidence assessment and quality tier classification")
        print("✅ Batch processing and statistical analysis")
        print("✅ Top paper identification and ranking")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\nDemo failed: {str(e)}")


if __name__ == "__main__":
    main()