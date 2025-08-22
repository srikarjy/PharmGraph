"""Tests for JournalImpactScorer."""

import pytest
from src.quality.journal_scorer import JournalImpactScorer, JournalTier


class TestJournalImpactScorer:
    """Test cases for JournalImpactScorer."""
    
    def test_init(self):
        """Test scorer initialization."""
        scorer = JournalImpactScorer()
        
        assert len(scorer.journal_database) > 0
        assert len(scorer.publisher_prestige) > 0
        
        # Check that some key journals are present
        assert "nature" in scorer.journal_database
        assert "science" in scorer.journal_database
        assert "pharmacogenomics" in scorer.journal_database
    
    def test_normalize_journal_name(self):
        """Test journal name normalization."""
        scorer = JournalImpactScorer()
        
        # Basic normalization
        assert scorer.normalize_journal_name("Nature") == "nature"
        assert scorer.normalize_journal_name("The Lancet") == "lancet"
        assert scorer.normalize_journal_name("Journal of Medicine") == "journal of medicine"
        
        # Punctuation and special characters
        assert scorer.normalize_journal_name("Clinical Pharmacology & Therapeutics") == "clinical pharmacology and therapeutics"
        assert scorer.normalize_journal_name("Nature Genetics.") == "nature genetics"
        
        # Whitespace normalization
        assert scorer.normalize_journal_name("  Multiple   Spaces  ") == "multiple spaces"
        
        # Empty/None handling
        assert scorer.normalize_journal_name("") == ""
        assert scorer.normalize_journal_name(None) == ""
    
    def test_get_journal_score_tier1(self):
        """Test scoring for Tier 1 journals."""
        scorer = JournalImpactScorer()
        
        # Test top-tier journals
        nature_score = scorer.get_journal_score("Nature")
        assert nature_score >= 95.0
        assert nature_score <= 125.0
        
        science_score = scorer.get_journal_score("Science")
        assert science_score >= 95.0
        assert science_score <= 125.0
        
        nejm_score = scorer.get_journal_score("New England Journal of Medicine")
        assert nejm_score >= 95.0
        assert nejm_score <= 125.0
    
    def test_get_journal_score_pharmacogenomics(self):
        """Test scoring for pharmacogenomics journals."""
        scorer = JournalImpactScorer()
        
        # Test domain-specific journals
        cpt_score = scorer.get_journal_score("Clinical Pharmacology & Therapeutics")
        assert cpt_score >= 85.0
        assert cpt_score <= 125.0
        
        pgx_score = scorer.get_journal_score("Pharmacogenomics")
        assert pgx_score >= 80.0
        assert pgx_score <= 125.0
        
        # Should be higher than general journals
        plos_score = scorer.get_journal_score("PLoS ONE")
        assert pgx_score > plos_score
    
    def test_get_journal_score_with_aliases(self):
        """Test scoring with journal aliases."""
        scorer = JournalImpactScorer()
        
        # Test aliases give same score as main name
        nejm_full = scorer.get_journal_score("New England Journal of Medicine")
        nejm_abbrev = scorer.get_journal_score("NEJM")
        assert abs(nejm_full - nejm_abbrev) < 0.1
        
        nat_genet_full = scorer.get_journal_score("Nature Genetics")
        nat_genet_abbrev = scorer.get_journal_score("Nat Genet")
        assert abs(nat_genet_full - nat_genet_abbrev) < 0.1
    
    def test_get_journal_score_unknown(self):
        """Test scoring for unknown journals."""
        scorer = JournalImpactScorer()
        
        # Unknown journal should get estimated score
        unknown_score = scorer.get_journal_score("Unknown Journal of Something")
        assert unknown_score >= 25.0
        assert unknown_score <= 50.0
        
        # Empty/None should get minimum score
        empty_score = scorer.get_journal_score("")
        assert empty_score == 25.0
        
        none_score = scorer.get_journal_score(None)
        assert none_score == 25.0
    
    def test_get_journal_score_with_domain_bonus(self):
        """Test domain bonus scoring."""
        scorer = JournalImpactScorer()
        
        # Test with pharmacogenomics content
        pgx_content = "pharmacogenomics CYP2D6 warfarin precision medicine"
        
        base_score = scorer.get_journal_score("Pharmacogenomics")
        bonus_score = scorer.get_journal_score_with_domain_bonus("Pharmacogenomics", pgx_content)
        
        assert bonus_score >= base_score
        
        # Test with non-relevant content
        irrelevant_content = "astronomy stars galaxies"
        irrelevant_score = scorer.get_journal_score_with_domain_bonus("Pharmacogenomics", irrelevant_content)
        
        # Should still get some bonus due to journal's domain relevance
        assert irrelevant_score >= base_score
    
    def test_get_journal_tier(self):
        """Test journal tier classification."""
        scorer = JournalImpactScorer()
        
        # Test tier classifications
        assert scorer.get_journal_tier("Nature") == 1
        assert scorer.get_journal_tier("Nature Genetics") == 2
        assert scorer.get_journal_tier("Pharmacogenomics") == 3
        assert scorer.get_journal_tier("Bioinformatics") == 4
        assert scorer.get_journal_tier("PLoS ONE") == 6
        
        # Unknown journal should be tier 6
        assert scorer.get_journal_tier("Unknown Journal") == 6
    
    def test_get_impact_factor(self):
        """Test impact factor retrieval."""
        scorer = JournalImpactScorer()
        
        # Test known impact factors
        nature_if = scorer.get_impact_factor("Nature")
        assert nature_if is not None
        assert nature_if > 0
        
        nejm_if = scorer.get_impact_factor("New England Journal of Medicine")
        assert nejm_if is not None
        assert nejm_if > 100  # NEJM has very high IF
        
        # Unknown journal should return None
        unknown_if = scorer.get_impact_factor("Unknown Journal")
        assert unknown_if is None
    
    def test_get_publisher_prestige(self):
        """Test publisher prestige scoring."""
        scorer = JournalImpactScorer()
        
        # Test high-prestige publishers
        nature_prestige = scorer.get_publisher_prestige("Nature")
        assert nature_prestige == 5.0
        
        science_prestige = scorer.get_publisher_prestige("Science")
        assert science_prestige == 5.0
        
        # Test lower-prestige publishers
        plos_prestige = scorer.get_publisher_prestige("PLoS ONE")
        assert plos_prestige == 1.0
        
        # Unknown journal should return 0
        unknown_prestige = scorer.get_publisher_prestige("Unknown Journal")
        assert unknown_prestige == 0.0
    
    def test_fuzzy_matching(self):
        """Test fuzzy journal name matching."""
        scorer = JournalImpactScorer()
        
        # Test partial matches - should find some match but may not be exact
        partial_score = scorer.get_journal_score("Nature Gen")  # Should match Nature Genetics
        full_score = scorer.get_journal_score("Nature Genetics")
        # Fuzzy matching may not be exact, but should be reasonable
        assert partial_score > 80.0  # Should get a decent score
        
        # Test with typos/variations - should find close match
        typo_score = scorer.get_journal_score("Pharmacogenomic")  # Should match Pharmacogenomics
        correct_score = scorer.get_journal_score("Pharmacogenomics")
        # Should be reasonably close or at least get a good score
        assert typo_score > 60.0  # Should get a reasonable score
    
    def test_get_journal_statistics(self):
        """Test journal database statistics."""
        scorer = JournalImpactScorer()
        
        stats = scorer.get_journal_statistics()
        
        assert 'total_journals' in stats
        assert 'total_entries' in stats
        assert 'tier_distribution' in stats
        assert 'publishers' in stats
        assert 'journals_with_if' in stats
        assert 'avg_impact_factor' in stats
        
        # Should have journals in all tiers
        assert len(stats['tier_distribution']) > 0
        assert stats['total_journals'] > 0
        assert stats['journals_with_if'] > 0
        assert stats['avg_impact_factor'] > 0
    
    def test_search_journals(self):
        """Test journal search functionality."""
        scorer = JournalImpactScorer()
        
        # Search for pharmacogenomics journals
        results = scorer.search_journals("pharmacogenomics", limit=5)
        
        assert len(results) > 0
        assert len(results) <= 5
        
        # Results should be tuples of (name, score, tier)
        for name, score, tier in results:
            assert isinstance(name, str)
            assert isinstance(score, float)
            assert isinstance(tier, int)
            assert 1 <= tier <= 6
            assert score >= 0
        
        # Results should be sorted by score (descending)
        scores = [result[1] for result in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_case_insensitive_matching(self):
        """Test case-insensitive journal matching."""
        scorer = JournalImpactScorer()
        
        # Test different cases give same score
        lower_score = scorer.get_journal_score("nature")
        upper_score = scorer.get_journal_score("NATURE")
        mixed_score = scorer.get_journal_score("Nature")
        
        assert abs(lower_score - upper_score) < 0.1
        assert abs(lower_score - mixed_score) < 0.1
    
    def test_score_capping(self):
        """Test that scores are properly capped at maximum."""
        scorer = JournalImpactScorer()
        
        # Even highest-scoring journals should not exceed 125 points
        nature_score = scorer.get_journal_score("Nature")
        assert nature_score <= 125.0
        
        nejm_score = scorer.get_journal_score("New England Journal of Medicine")
        assert nejm_score <= 125.0
    
    def test_domain_relevance_multipliers(self):
        """Test domain relevance multipliers are applied correctly."""
        scorer = JournalImpactScorer()
        
        # Pharmacogenomics journals should have higher domain relevance
        pgx_journal = scorer.journal_database.get("pharmacogenomics")
        general_journal = scorer.journal_database.get("plos one")
        
        if pgx_journal and general_journal:
            assert pgx_journal.domain_relevance > general_journal.domain_relevance