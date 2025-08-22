"""Tests for PubMedXMLParser."""

import pytest
from datetime import datetime

from src.data_ingestion.pubmed_parser import PubMedXMLParser
from src.data_ingestion.data_models import PaperMetadata, Author, MeshTerm, Journal


class TestPubMedXMLParser:
    """Test cases for PubMedXMLParser."""
    
    def test_init(self):
        """Test parser initialization."""
        parser = PubMedXMLParser()
        assert parser.namespace_map is not None
    
    def test_parse_pubmed_xml_empty(self):
        """Test parsing empty XML."""
        parser = PubMedXMLParser()
        
        papers = parser.parse_pubmed_xml("")
        assert papers == []
        
        papers = parser.parse_pubmed_xml("   ")
        assert papers == []
    
    def test_parse_pubmed_xml_invalid(self):
        """Test parsing invalid XML."""
        parser = PubMedXMLParser()
        
        papers = parser.parse_pubmed_xml("<invalid>xml")
        assert papers == []
    
    def test_parse_pubmed_xml_single_article(self):
        """Test parsing single article XML."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345</PMID>
                    <Article>
                        <ArticleTitle>Test Article Title</ArticleTitle>
                        <Abstract>
                            <AbstractText>This is a test abstract.</AbstractText>
                        </Abstract>
                        <AuthorList>
                            <Author>
                                <LastName>Smith</LastName>
                                <ForeName>John</ForeName>
                                <Initials>J</Initials>
                            </Author>
                        </AuthorList>
                        <Journal>
                            <Title>Test Journal</Title>
                            <ISOAbbreviation>Test J</ISOAbbreviation>
                        </Journal>
                        <PublicationTypeList>
                            <PublicationType>Journal Article</PublicationType>
                        </PublicationTypeList>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        
        assert len(papers) == 1
        paper = papers[0]
        assert paper.pmid == "12345"
        assert paper.title == "Test Article Title"
        assert paper.abstract == "This is a test abstract."
        assert len(paper.authors) == 1
        assert paper.authors[0].last_name == "Smith"
        assert paper.authors[0].first_name == "John"
        assert paper.journal.title == "Test Journal"
        assert "Journal Article" in paper.publication_types
    
    def test_parse_pubmed_xml_multiple_articles(self):
        """Test parsing multiple articles."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345</PMID>
                    <Article>
                        <ArticleTitle>First Article</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>67890</PMID>
                    <Article>
                        <ArticleTitle>Second Article</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        
        assert len(papers) == 2
        assert papers[0].pmid == "12345"
        assert papers[0].title == "First Article"
        assert papers[1].pmid == "67890"
        assert papers[1].title == "Second Article"
    
    def test_extract_pmid(self):
        """Test PMID extraction."""
        parser = PubMedXMLParser()
        
        # Test standard PMID location
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
            </MedlineCitation>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        assert len(papers) == 1
        assert papers[0].pmid == "12345"
    
    def test_extract_title(self):
        """Test title extraction."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Pharmacogenomics and Machine Learning: A Review</ArticleTitle>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        assert papers[0].title == "Pharmacogenomics and Machine Learning: A Review"
    
    def test_extract_abstract_simple(self):
        """Test simple abstract extraction."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Test</ArticleTitle>
                    <Abstract>
                        <AbstractText>This is a simple abstract.</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        assert papers[0].abstract == "This is a simple abstract."
    
    def test_extract_abstract_structured(self):
        """Test structured abstract extraction."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Test</ArticleTitle>
                    <Abstract>
                        <AbstractText Label="BACKGROUND">Background information.</AbstractText>
                        <AbstractText Label="METHODS">Methods used.</AbstractText>
                        <AbstractText Label="RESULTS">Results obtained.</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        paper = papers[0]
        
        assert "Background information. Methods used. Results obtained." in paper.abstract
        assert len(paper.abstract_sections) == 3
        assert paper.abstract_sections[0].label == "BACKGROUND"
        assert paper.abstract_sections[0].text == "Background information."
        assert paper.get_abstract_section("METHODS") == "Methods used."
    
    def test_extract_authors(self):
        """Test author extraction."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Test</ArticleTitle>
                    <AuthorList>
                        <Author>
                            <LastName>Smith</LastName>
                            <ForeName>John</ForeName>
                            <Initials>J</Initials>
                            <AffiliationInfo>
                                <Affiliation>University of Test</Affiliation>
                            </AffiliationInfo>
                        </Author>
                        <Author>
                            <LastName>Doe</LastName>
                            <ForeName>Jane</ForeName>
                            <Initials>JD</Initials>
                        </Author>
                    </AuthorList>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        authors = papers[0].authors
        
        assert len(authors) == 2
        
        assert authors[0].last_name == "Smith"
        assert authors[0].first_name == "John"
        assert authors[0].initials == "J"
        assert authors[0].affiliation == "University of Test"
        
        assert authors[1].last_name == "Doe"
        assert authors[1].first_name == "Jane"
        assert authors[1].initials == "JD"
        assert authors[1].affiliation is None
    
    def test_extract_mesh_terms(self):
        """Test MeSH term extraction."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Test</ArticleTitle>
                </Article>
                <MeshHeadingList>
                    <MeshHeading>
                        <DescriptorName UI="D010597" MajorTopicYN="Y">Pharmacogenetics</DescriptorName>
                    </MeshHeading>
                    <MeshHeading>
                        <DescriptorName UI="D000069550" MajorTopicYN="N">Machine Learning</DescriptorName>
                        <QualifierName UI="Q000379" MajorTopicYN="Y">methods</QualifierName>
                    </MeshHeading>
                </MeshHeadingList>
            </MedlineCitation>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        mesh_terms = papers[0].mesh_terms
        
        assert len(mesh_terms) == 2
        
        assert mesh_terms[0].descriptor_name == "Pharmacogenetics"
        assert mesh_terms[0].descriptor_ui == "D010597"
        assert mesh_terms[0].major_topic is True
        assert mesh_terms[0].qualifier_name is None
        
        assert mesh_terms[1].descriptor_name == "Machine Learning"
        assert mesh_terms[1].descriptor_ui == "D000069550"
        assert mesh_terms[1].major_topic is False
        assert mesh_terms[1].qualifier_name == "methods"
    
    def test_extract_publication_date(self):
        """Test publication date extraction."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Test</ArticleTitle>
                    <Journal>
                        <JournalIssue>
                            <PubDate>
                                <Year>2023</Year>
                                <Month>Mar</Month>
                                <Day>15</Day>
                            </PubDate>
                        </JournalIssue>
                    </Journal>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        pub_date = papers[0].publication_date
        
        assert pub_date is not None
        assert pub_date.year == 2023
        assert pub_date.month == 3
        assert pub_date.day == 15
    
    def test_extract_journal(self):
        """Test journal extraction."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <Journal>
                        <ISSN IssnType="Electronic">1234-5678</ISSN>
                        <JournalIssue CitedMedium="Internet">
                            <Volume>25</Volume>
                            <Issue>3</Issue>
                            <PubDate>
                                <Year>2023</Year>
                            </PubDate>
                        </JournalIssue>
                        <Title>Journal of Pharmacogenomics</Title>
                        <ISOAbbreviation>J Pharmacogenomics</ISOAbbreviation>
                    </Journal>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        journal = papers[0].journal
        
        assert journal is not None
        assert journal.title == "Journal of Pharmacogenomics"
        assert journal.iso_abbreviation == "J Pharmacogenomics"
        assert journal.issn == "1234-5678"
        assert journal.issn_type == "Electronic"
        assert journal.volume == "25"
        assert journal.issue == "3"
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Test</ArticleTitle>
                </Article>
                <KeywordList>
                    <Keyword MajorTopicYN="N">pharmacogenomics</Keyword>
                    <Keyword MajorTopicYN="Y">machine learning</Keyword>
                    <Keyword MajorTopicYN="N">precision medicine</Keyword>
                </KeywordList>
            </MedlineCitation>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        keywords = papers[0].keywords
        
        assert len(keywords) == 3
        assert "pharmacogenomics" in keywords
        assert "machine learning" in keywords
        assert "precision medicine" in keywords
    
    def test_extract_publication_types(self):
        """Test publication type extraction."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Test</ArticleTitle>
                    <PublicationTypeList>
                        <PublicationType UI="D016428">Journal Article</PublicationType>
                        <PublicationType UI="D016454">Review</PublicationType>
                    </PublicationTypeList>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        pub_types = papers[0].publication_types
        
        assert len(pub_types) == 2
        assert "Journal Article" in pub_types
        assert "Review" in pub_types
    
    def test_extract_doi_and_pmc(self):
        """Test DOI and PMC ID extraction."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Test</ArticleTitle>
                </Article>
            </MedlineCitation>
            <PubmedData>
                <ArticleIdList>
                    <ArticleId IdType="pubmed">12345</ArticleId>
                    <ArticleId IdType="doi">10.1000/test.doi</ArticleId>
                    <ArticleId IdType="pmc">PMC1234567</ArticleId>
                </ArticleIdList>
            </PubmedData>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        paper = papers[0]
        
        assert paper.doi == "10.1000/test.doi"
        assert paper.pmc_id == "PMC1234567"
    
    def test_extract_grants(self):
        """Test grant extraction."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Test</ArticleTitle>
                    <GrantList>
                        <Grant>
                            <GrantID>R01GM123456</GrantID>
                            <Acronym>GM</Acronym>
                            <Agency>NIGMS NIH HHS</Agency>
                            <Country>United States</Country>
                        </Grant>
                        <Grant>
                            <GrantID>K99CA987654</GrantID>
                            <Agency>NCI NIH HHS</Agency>
                        </Grant>
                    </GrantList>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        grants = papers[0].grants
        
        assert len(grants) == 2
        
        assert grants[0].grant_id == "R01GM123456"
        assert grants[0].acronym == "GM"
        assert grants[0].agency == "NIGMS NIH HHS"
        assert grants[0].country == "United States"
        
        assert grants[1].grant_id == "K99CA987654"
        assert grants[1].agency == "NCI NIH HHS"
        assert grants[1].acronym is None
    
    def test_parse_month_numeric(self):
        """Test parsing numeric month."""
        parser = PubMedXMLParser()
        
        assert parser._parse_month("3") == 3
        assert parser._parse_month("12") == 12
    
    def test_parse_month_text(self):
        """Test parsing text month."""
        parser = PubMedXMLParser()
        
        assert parser._parse_month("Mar") == 3
        assert parser._parse_month("December") == 12
        assert parser._parse_month("Invalid") == 1  # Default
    
    def test_clean_text(self):
        """Test text cleaning."""
        parser = PubMedXMLParser()
        
        # Test whitespace normalization
        assert parser._clean_text("  multiple   spaces  ") == "multiple spaces"
        
        # Test HTML entity handling
        assert parser._clean_text("A &amp; B &lt; C") == "A & B < C"
        
        # Test empty text
        assert parser._clean_text("") == ""
        assert parser._clean_text(None) == ""
    
    def test_validate_paper_metadata_valid(self):
        """Test validation of valid paper metadata."""
        parser = PubMedXMLParser()
        
        paper = PaperMetadata(
            pmid="12345",
            title="Test Paper",
            authors=[Author(last_name="Smith")]
        )
        
        assert parser.validate_paper_metadata(paper) is True
    
    def test_validate_paper_metadata_invalid(self):
        """Test validation of invalid paper metadata."""
        parser = PubMedXMLParser()
        
        # Missing PMID
        paper = PaperMetadata(pmid="", title="Test Paper")
        assert parser.validate_paper_metadata(paper) is False
        
        # Missing title
        paper = PaperMetadata(pmid="12345", title="")
        assert parser.validate_paper_metadata(paper) is False
    
    def test_parse_malformed_article(self):
        """Test parsing with malformed article (should skip and continue)."""
        parser = PubMedXMLParser()
        
        xml_content = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345</PMID>
                    <Article>
                        <ArticleTitle>Good Article</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
            <PubmedArticle>
                <!-- Malformed article - missing PMID -->
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Bad Article</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>67890</PMID>
                    <Article>
                        <ArticleTitle>Another Good Article</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        
        papers = parser.parse_pubmed_xml(xml_content)
        
        # Should get 2 papers (skip the malformed one)
        assert len(papers) == 2
        assert papers[0].pmid == "12345"
        assert papers[1].pmid == "67890"