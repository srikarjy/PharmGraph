"""Tests for PharmacogenomicsQueryBuilder."""

import pytest
from src.data_ingestion.pgx_queries import (
    PharmacogenomicsQueryBuilder, QueryTemplate, QueryComponents
)


class TestPharmacogenomicsQueryBuilder:
    """Test cases for PharmacogenomicsQueryBuilder."""
    
    def test_init(self):
        """Test initialization."""
        builder = PharmacogenomicsQueryBuilder()
        
        assert len(builder.pgx_core_terms) > 0
        assert len(builder.ml_ai_terms) > 0
        assert len(builder.pharmacogenes) > 0
        assert len(builder.drug_categories) > 0
        assert len(builder.study_types) > 0
        assert len(builder.high_impact_journals) > 0
        
        # Check specific terms are present
        assert "pharmacogenomics" in builder.pgx_core_terms
        assert "machine learning" in builder.ml_ai_terms
        assert "CYP2D6" in builder.pharmacogenes
        assert "warfarin" in builder.drug_categories["anticoagulants"]
    
    def test_build_pgx_ml_query_basic(self):
        """Test building basic PGx-ML query."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_pgx_ml_query()
        
        assert "pharmacogenomics" in query.lower()
        assert "precision medicine" in query.lower()
        assert "machine learning" in query.lower()
        assert "artificial intelligence" in query.lower()
        assert "MeSH" in query
        assert "Title/Abstract" in query
    
    def test_build_pgx_ml_query_with_drugs(self):
        """Test building PGx-ML query with specific drugs."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_pgx_ml_query(
            drug_terms=["warfarin", "clopidogrel"],
            gene_terms=["CYP2D6", "VKORC1"]
        )
        
        assert "warfarin" in query
        assert "clopidogrel" in query
        assert "CYP2D6" in query
        assert "VKORC1" in query
    
    def test_build_pgx_ml_query_with_date_range(self):
        """Test building PGx-ML query with date range."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_pgx_ml_query(
            date_range=("2020/01/01", "2023/12/31")
        )
        
        assert "2020/01/01" in query
        assert "2023/12/31" in query
        assert "Date - Publication" in query
    
    def test_build_pgx_ml_query_without_mesh(self):
        """Test building PGx-ML query without MeSH terms."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_pgx_ml_query(use_mesh=False)
        
        assert "MeSH" not in query
        assert "Title/Abstract" in query
    
    def test_build_cyp_ai_query_basic(self):
        """Test building basic CYP-AI query."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_cyp_ai_query()
        
        assert "CYP2D6" in query
        assert "CYP2C19" in query
        assert "artificial intelligence" in query
        assert "deep learning" in query
        assert "substrate" in query
        assert "metabolism" in query
    
    def test_build_cyp_ai_query_custom(self):
        """Test building CYP-AI query with custom parameters."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_cyp_ai_query(
            cyp_enzymes=["CYP3A4"],
            ai_terms=["neural network"],
            include_substrates=False
        )
        
        assert "CYP3A4" in query
        assert "neural network" in query
        assert "substrate" not in query
    
    def test_build_drug_dosing_query_basic(self):
        """Test building basic drug dosing query."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_drug_dosing_query()
        
        assert "warfarin" in query
        assert "dosing algorithm" in query
        assert "dose prediction" in query
    
    def test_build_drug_dosing_query_custom(self):
        """Test building drug dosing query with custom parameters."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_drug_dosing_query(
            drugs=["tacrolimus"],
            algorithm_types=["dose optimization"]
        )
        
        assert "tacrolimus" in query
        assert "dose optimization" in query
    
    def test_build_precision_medicine_query_basic(self):
        """Test building basic precision medicine query."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_precision_medicine_query()
        
        assert "precision medicine" in query
        assert "MeSH" in query
        assert "drug interaction" in query
        assert "biomarker" in query
    
    def test_build_precision_medicine_query_adverse_events(self):
        """Test building precision medicine query for adverse events."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_precision_medicine_query(
            focus_area="adverse_events",
            include_biomarkers=False
        )
        
        assert "adverse drug reaction" in query
        assert "side effect" in query
        assert "biomarker" not in query
    
    def test_build_pharmacokinetics_ml_query_basic(self):
        """Test building basic PK-ML query."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_pharmacokinetics_ml_query()
        
        assert "pharmacokinetics" in query
        assert "MeSH" in query
        assert "clearance" in query
        assert "neural network" in query
    
    def test_build_pharmacokinetics_ml_query_custom(self):
        """Test building PK-ML query with custom parameters."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_pharmacokinetics_ml_query(
            pk_parameters=["Cmax", "AUC"],
            ml_methods=["random forest"]
        )
        
        assert "Cmax" in query
        assert "AUC" in query
        assert "random forest" in query
    
    def test_build_clinical_trial_pgx_query_basic(self):
        """Test building basic clinical trial PGx query."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_clinical_trial_pgx_query()
        
        assert "clinical trial" in query
        assert "Publication Type" in query
        assert "pharmacogenomics" in query
        assert "MeSH" in query
    
    def test_build_clinical_trial_pgx_query_with_phase(self):
        """Test building clinical trial PGx query with phase."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_clinical_trial_pgx_query(phase="II")
        
        assert "phase II" in query
    
    def test_build_adverse_reactions_ml_query_basic(self):
        """Test building basic ADR-ML query."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_adverse_reactions_ml_query()
        
        assert "adverse drug reaction" in query
        assert "hepatotoxicity" in query
        assert "machine learning" in query
        assert "prediction" in query
    
    def test_build_adverse_reactions_ml_query_custom(self):
        """Test building ADR-ML query with custom parameters."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_adverse_reactions_ml_query(
            reaction_types=["cardiotoxicity"],
            prediction_focus=False
        )
        
        assert "cardiotoxicity" in query
        # When prediction_focus=False, specific prediction terms shouldn't be added
        assert "risk assessment" not in query
        assert "early detection" not in query
    
    def test_build_template_query_pgx_ml_basic(self):
        """Test building query from PGX_ML_BASIC template."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_template_query(QueryTemplate.PGX_ML_BASIC)
        
        assert "pharmacogenomics" in query.lower()
        assert "machine learning" in query.lower()
    
    def test_build_template_query_cyp_ai(self):
        """Test building query from CYP_AI template."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_template_query(QueryTemplate.CYP_AI)
        
        assert "CYP" in query
        assert "artificial intelligence" in query.lower()
    
    def test_build_template_query_with_params(self):
        """Test building query from template with custom parameters."""
        builder = PharmacogenomicsQueryBuilder()
        
        query = builder.build_template_query(
            QueryTemplate.DRUG_DOSING,
            custom_params={"drugs": ["metoprolol"]}
        )
        
        assert "metoprolol" in query
    
    def test_build_template_query_unknown(self):
        """Test building query from unknown template."""
        builder = PharmacogenomicsQueryBuilder()
        
        # Create a mock unknown template
        class UnknownTemplate:
            pass
        
        query = builder.build_template_query(UnknownTemplate())
        
        assert query == ""
    
    def test_get_high_impact_journal_filter_default(self):
        """Test getting high-impact journal filter with defaults."""
        builder = PharmacogenomicsQueryBuilder()
        
        filter_str = builder.get_high_impact_journal_filter()
        
        assert "Nature" in filter_str
        assert "Science" in filter_str
        assert "Journal" in filter_str
        assert filter_str.startswith("(")
        assert filter_str.endswith(")")
    
    def test_get_high_impact_journal_filter_custom(self):
        """Test getting high-impact journal filter with custom journals."""
        builder = PharmacogenomicsQueryBuilder()
        
        filter_str = builder.get_high_impact_journal_filter(["Custom Journal"])
        
        assert "Custom Journal" in filter_str
        assert "Nature" not in filter_str
    
    def test_get_recent_years_filter(self):
        """Test getting recent years filter."""
        builder = PharmacogenomicsQueryBuilder()
        
        filter_str = builder.get_recent_years_filter(years_back=3)
        
        assert "Date - Publication" in filter_str
        # Should contain a year range
        assert ":" in filter_str
    
    def test_combine_queries_single(self):
        """Test combining single query."""
        builder = PharmacogenomicsQueryBuilder()
        
        combined = builder.combine_queries(["query1"])
        
        assert combined == "query1"
    
    def test_combine_queries_multiple_or(self):
        """Test combining multiple queries with OR."""
        builder = PharmacogenomicsQueryBuilder()
        
        combined = builder.combine_queries(["query1", "query2", "query3"], operator="OR")
        
        assert combined == "(query1) OR (query2) OR (query3)"
    
    def test_combine_queries_multiple_and(self):
        """Test combining multiple queries with AND."""
        builder = PharmacogenomicsQueryBuilder()
        
        combined = builder.combine_queries(["query1", "query2"], operator="AND")
        
        assert combined == "(query1) AND (query2)"
    
    def test_combine_queries_empty(self):
        """Test combining empty queries."""
        builder = PharmacogenomicsQueryBuilder()
        
        combined = builder.combine_queries([])
        
        assert combined == ""
    
    def test_get_available_templates(self):
        """Test getting available templates."""
        builder = PharmacogenomicsQueryBuilder()
        
        templates = builder.get_available_templates()
        
        assert len(templates) > 0
        assert "pharmacogenomics_ml_basic" in templates
        assert "cyp_artificial_intelligence" in templates
    
    def test_get_pharmacogenes_by_category_cyp(self):
        """Test getting CYP genes."""
        builder = PharmacogenomicsQueryBuilder()
        
        cyp_genes = builder.get_pharmacogenes_by_category("cyp")
        
        assert "CYP2D6" in cyp_genes
        assert "CYP2C19" in cyp_genes
        assert "CYP3A4" in cyp_genes
    
    def test_get_pharmacogenes_by_category_transporter(self):
        """Test getting transporter genes."""
        builder = PharmacogenomicsQueryBuilder()
        
        transporter_genes = builder.get_pharmacogenes_by_category("transporter")
        
        assert "SLCO1B1" in transporter_genes
        assert "ABCB1" in transporter_genes
    
    def test_get_pharmacogenes_by_category_unknown(self):
        """Test getting genes for unknown category."""
        builder = PharmacogenomicsQueryBuilder()
        
        unknown_genes = builder.get_pharmacogenes_by_category("unknown")
        
        assert unknown_genes == []