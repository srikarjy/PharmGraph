"""PubMed XML parser for extracting structured paper metadata."""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional, Dict, Any
import re

from ..config import get_logger
from .data_models import (
    PaperMetadata, Author, MeshTerm, Grant, Journal, 
    AbstractSection, PublicationType
)


logger = get_logger(__name__)


class PubMedXMLParser:
    """Parser for PubMed XML responses."""
    
    def __init__(self):
        """Initialize the parser."""
        self.namespace_map = {
            'pubmed': 'https://www.ncbi.nlm.nih.gov/pubmed/',
            'medline': 'https://www.ncbi.nlm.nih.gov/medline/'
        }
        
        logger.info("Initialized PubMedXMLParser")
    
    def parse_pubmed_xml(self, xml_content: str) -> List[PaperMetadata]:
        """Parse PubMed XML response into structured paper metadata.
        
        Args:
            xml_content: Raw XML content from NCBI efetch
            
        Returns:
            List[PaperMetadata]: List of parsed paper metadata
        """
        if not xml_content or not xml_content.strip():
            logger.warning("Empty XML content provided")
            return []
        
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Handle different root elements
            if root.tag == 'PubmedArticleSet':
                articles = root.findall('.//PubmedArticle')
            elif root.tag == 'PubmedArticle':
                articles = [root]
            else:
                # Try to find PubmedArticle elements anywhere in the tree
                articles = root.findall('.//PubmedArticle')
                if not articles:
                    logger.error(f"Unexpected XML root element: {root.tag}")
                    return []
            
            papers = []
            for article in articles:
                try:
                    paper = self.extract_paper_from_article(article)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    # Log error but continue with other articles
                    pmid = self._extract_pmid(article)
                    logger.error(f"Failed to parse article {pmid}: {str(e)}")
                    continue
            
            logger.info(f"Successfully parsed {len(papers)} papers from XML")
            return papers
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing XML: {str(e)}")
            return []
    
    def extract_paper_from_article(self, article_elem: ET.Element) -> Optional[PaperMetadata]:
        """Extract paper metadata from a PubmedArticle element.
        
        Args:
            article_elem: PubmedArticle XML element
            
        Returns:
            Optional[PaperMetadata]: Parsed paper metadata
        """
        try:
            # Extract PMID (required)
            pmid = self._extract_pmid(article_elem)
            if not pmid:
                logger.warning("Article missing PMID, skipping")
                return None
            
            # Create paper metadata object
            paper = PaperMetadata(pmid=pmid)
            
            # Extract basic information
            paper.title = self._extract_title(article_elem)
            paper.abstract = self._extract_abstract(article_elem)
            paper.abstract_sections = self._extract_abstract_sections(article_elem)
            
            # Extract identifiers
            paper.pmc_id = self._extract_pmc_id(article_elem)
            paper.doi = self._extract_doi(article_elem)
            
            # Extract authors
            paper.authors = self.extract_authors(article_elem)
            
            # Extract journal information
            paper.journal = self._extract_journal(article_elem)
            
            # Extract dates
            paper.publication_date = self.extract_publication_date(article_elem)
            paper.epub_date = self._extract_epub_date(article_elem)
            
            # Extract classification
            paper.mesh_terms = self.extract_mesh_terms(article_elem)
            paper.keywords = self._extract_keywords(article_elem)
            paper.publication_types = self._extract_publication_types(article_elem)
            
            # Extract additional metadata
            paper.language = self._extract_language(article_elem)
            paper.country = self._extract_country(article_elem)
            paper.grants = self._extract_grants(article_elem)
            paper.pages = self._extract_pages(article_elem)
            paper.elocation_id = self._extract_elocation_id(article_elem)
            
            # Store original XML for debugging
            paper.source_xml = ET.tostring(article_elem, encoding='unicode')
            
            return paper
            
        except Exception as e:
            logger.error(f"Error extracting paper metadata: {str(e)}")
            return None
    
    def _extract_pmid(self, article_elem: ET.Element) -> Optional[str]:
        """Extract PMID from article element."""
        # Try different possible locations for PMID
        pmid_elem = article_elem.find('.//PMID')
        if pmid_elem is not None and pmid_elem.text:
            return pmid_elem.text.strip()
        
        # Alternative location
        pmid_elem = article_elem.find('.//ArticleId[@IdType="pubmed"]')
        if pmid_elem is not None and pmid_elem.text:
            return pmid_elem.text.strip()
        
        return None
    
    def _extract_title(self, article_elem: ET.Element) -> str:
        """Extract article title."""
        title_elem = article_elem.find('.//ArticleTitle')
        if title_elem is not None:
            # Handle HTML entities and formatting
            title = ET.tostring(title_elem, encoding='unicode', method='text')
            return self._clean_text(title)
        return ""
    
    def _extract_abstract(self, article_elem: ET.Element) -> str:
        """Extract complete abstract text."""
        abstract_parts = []
        
        # Look for structured abstract
        abstract_texts = article_elem.findall('.//AbstractText')
        if abstract_texts:
            for abstract_text in abstract_texts:
                text = ET.tostring(abstract_text, encoding='unicode', method='text')
                if text.strip():
                    abstract_parts.append(self._clean_text(text))
        else:
            # Look for simple abstract
            abstract_elem = article_elem.find('.//Abstract')
            if abstract_elem is not None:
                text = ET.tostring(abstract_elem, encoding='unicode', method='text')
                if text.strip():
                    abstract_parts.append(self._clean_text(text))
        
        return " ".join(abstract_parts)
    
    def _extract_abstract_sections(self, article_elem: ET.Element) -> List[AbstractSection]:
        """Extract structured abstract sections."""
        sections = []
        
        abstract_texts = article_elem.findall('.//AbstractText')
        for abstract_text in abstract_texts:
            label = abstract_text.get('Label')
            text = ET.tostring(abstract_text, encoding='unicode', method='text')
            
            if text.strip():
                section = AbstractSection(
                    label=label,
                    text=self._clean_text(text)
                )
                sections.append(section)
        
        return sections
    
    def extract_authors(self, article_elem: ET.Element) -> List[Author]:
        """Extract author information."""
        authors = []
        
        author_list = article_elem.find('.//AuthorList')
        if author_list is not None:
            for author_elem in author_list.findall('Author'):
                author = self._extract_single_author(author_elem)
                if author:
                    authors.append(author)
        
        return authors
    
    def _extract_single_author(self, author_elem: ET.Element) -> Optional[Author]:
        """Extract single author information."""
        try:
            # Extract name components
            last_name_elem = author_elem.find('LastName')
            first_name_elem = author_elem.find('ForeName')
            initials_elem = author_elem.find('Initials')
            suffix_elem = author_elem.find('Suffix')
            
            last_name = last_name_elem.text if last_name_elem is not None else None
            first_name = first_name_elem.text if first_name_elem is not None else None
            initials = initials_elem.text if initials_elem is not None else None
            suffix = suffix_elem.text if suffix_elem is not None else None
            
            if not last_name:
                return None
            
            # Extract affiliation
            affiliation = None
            affiliation_elem = author_elem.find('.//Affiliation')
            if affiliation_elem is not None and affiliation_elem.text:
                affiliation = self._clean_text(affiliation_elem.text)
            
            # Extract ORCID if available
            orcid = None
            identifier_elem = author_elem.find('.//Identifier[@Source="ORCID"]')
            if identifier_elem is not None and identifier_elem.text:
                orcid = identifier_elem.text.strip()
            
            return Author(
                last_name=last_name,
                first_name=first_name,
                initials=initials,
                suffix=suffix,
                affiliation=affiliation,
                orcid=orcid
            )
            
        except Exception as e:
            logger.warning(f"Error extracting author: {str(e)}")
            return None
    
    def extract_mesh_terms(self, article_elem: ET.Element) -> List[MeshTerm]:
        """Extract MeSH terms."""
        mesh_terms = []
        
        mesh_heading_list = article_elem.find('.//MeshHeadingList')
        if mesh_heading_list is not None:
            for mesh_heading in mesh_heading_list.findall('MeshHeading'):
                mesh_term = self._extract_single_mesh_term(mesh_heading)
                if mesh_term:
                    mesh_terms.append(mesh_term)
        
        return mesh_terms
    
    def _extract_single_mesh_term(self, mesh_heading_elem: ET.Element) -> Optional[MeshTerm]:
        """Extract single MeSH term."""
        try:
            descriptor_elem = mesh_heading_elem.find('DescriptorName')
            if descriptor_elem is None:
                return None
            
            descriptor_name = descriptor_elem.text
            descriptor_ui = descriptor_elem.get('UI')
            major_topic = descriptor_elem.get('MajorTopicYN') == 'Y'
            
            # Check for qualifier
            qualifier_name = None
            qualifier_ui = None
            qualifier_elem = mesh_heading_elem.find('QualifierName')
            if qualifier_elem is not None:
                qualifier_name = qualifier_elem.text
                qualifier_ui = qualifier_elem.get('UI')
            
            return MeshTerm(
                descriptor_name=descriptor_name,
                descriptor_ui=descriptor_ui,
                qualifier_name=qualifier_name,
                qualifier_ui=qualifier_ui,
                major_topic=major_topic
            )
            
        except Exception as e:
            logger.warning(f"Error extracting MeSH term: {str(e)}")
            return None
    
    def extract_publication_date(self, article_elem: ET.Element) -> Optional[datetime]:
        """Extract publication date."""
        # Try different date elements in order of preference
        date_elements = [
            './/PubDate',
            './/ArticleDate[@DateType="Electronic"]',
            './/DateCompleted',
            './/DateRevised'
        ]
        
        for date_path in date_elements:
            date_elem = article_elem.find(date_path)
            if date_elem is not None:
                date = self._parse_date_element(date_elem)
                if date:
                    return date
        
        return None
    
    def _parse_date_element(self, date_elem: ET.Element) -> Optional[datetime]:
        """Parse date from XML element."""
        try:
            year_elem = date_elem.find('Year')
            month_elem = date_elem.find('Month')
            day_elem = date_elem.find('Day')
            
            year = int(year_elem.text) if year_elem is not None and year_elem.text else None
            month = self._parse_month(month_elem.text) if month_elem is not None and month_elem.text else 1
            day = int(day_elem.text) if day_elem is not None and day_elem.text else 1
            
            if year:
                return datetime(year, month, day)
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Error parsing date: {str(e)}")
        
        return None
    
    def _parse_month(self, month_str: str) -> int:
        """Parse month from string (handles both numeric and text months)."""
        if month_str.isdigit():
            return int(month_str)
        
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        return month_map.get(month_str, 1)
    
    def _extract_journal(self, article_elem: ET.Element) -> Optional[Journal]:
        """Extract journal information."""
        journal_elem = article_elem.find('.//Journal')
        if journal_elem is None:
            return None
        
        try:
            title_elem = journal_elem.find('Title')
            iso_abbrev_elem = journal_elem.find('ISOAbbreviation')
            issn_elem = journal_elem.find('ISSN')
            
            title = title_elem.text if title_elem is not None else None
            iso_abbreviation = iso_abbrev_elem.text if iso_abbrev_elem is not None else None
            issn = issn_elem.text if issn_elem is not None else None
            issn_type = issn_elem.get('IssnType') if issn_elem is not None else None
            
            # Extract volume and issue
            journal_issue = journal_elem.find('JournalIssue')
            volume = None
            issue = None
            pub_date = None
            
            if journal_issue is not None:
                volume_elem = journal_issue.find('Volume')
                issue_elem = journal_issue.find('Issue')
                pub_date_elem = journal_issue.find('PubDate')
                
                volume = volume_elem.text if volume_elem is not None else None
                issue = issue_elem.text if issue_elem is not None else None
                
                if pub_date_elem is not None:
                    pub_date = self._parse_date_element(pub_date_elem)
            
            if title:
                return Journal(
                    title=title,
                    iso_abbreviation=iso_abbreviation,
                    issn=issn,
                    issn_type=issn_type,
                    volume=volume,
                    issue=issue,
                    pub_date=pub_date
                )
            
        except Exception as e:
            logger.warning(f"Error extracting journal: {str(e)}")
        
        return None
    
    def _extract_pmc_id(self, article_elem: ET.Element) -> Optional[str]:
        """Extract PMC ID."""
        pmc_elem = article_elem.find('.//ArticleId[@IdType="pmc"]')
        if pmc_elem is not None and pmc_elem.text:
            return pmc_elem.text.strip()
        return None
    
    def _extract_doi(self, article_elem: ET.Element) -> Optional[str]:
        """Extract DOI."""
        doi_elem = article_elem.find('.//ArticleId[@IdType="doi"]')
        if doi_elem is not None and doi_elem.text:
            return doi_elem.text.strip()
        return None
    
    def _extract_keywords(self, article_elem: ET.Element) -> List[str]:
        """Extract author keywords."""
        keywords = []
        
        keyword_list = article_elem.find('.//KeywordList')
        if keyword_list is not None:
            for keyword_elem in keyword_list.findall('Keyword'):
                if keyword_elem.text:
                    keywords.append(self._clean_text(keyword_elem.text))
        
        return keywords
    
    def _extract_publication_types(self, article_elem: ET.Element) -> List[str]:
        """Extract publication types."""
        pub_types = []
        
        pub_type_list = article_elem.find('.//PublicationTypeList')
        if pub_type_list is not None:
            for pub_type_elem in pub_type_list.findall('PublicationType'):
                if pub_type_elem.text:
                    pub_types.append(pub_type_elem.text.strip())
        
        return pub_types
    
    def _extract_language(self, article_elem: ET.Element) -> str:
        """Extract language."""
        language_elem = article_elem.find('.//Language')
        if language_elem is not None and language_elem.text:
            return language_elem.text.strip()
        return "eng"  # Default to English
    
    def _extract_country(self, article_elem: ET.Element) -> Optional[str]:
        """Extract country of publication."""
        country_elem = article_elem.find('.//Country')
        if country_elem is not None and country_elem.text:
            return country_elem.text.strip()
        return None
    
    def _extract_grants(self, article_elem: ET.Element) -> List[Grant]:
        """Extract grant information."""
        grants = []
        
        grant_list = article_elem.find('.//GrantList')
        if grant_list is not None:
            for grant_elem in grant_list.findall('Grant'):
                grant = self._extract_single_grant(grant_elem)
                if grant:
                    grants.append(grant)
        
        return grants
    
    def _extract_single_grant(self, grant_elem: ET.Element) -> Optional[Grant]:
        """Extract single grant information."""
        try:
            grant_id_elem = grant_elem.find('GrantID')
            acronym_elem = grant_elem.find('Acronym')
            agency_elem = grant_elem.find('Agency')
            country_elem = grant_elem.find('Country')
            
            grant_id = grant_id_elem.text if grant_id_elem is not None else None
            acronym = acronym_elem.text if acronym_elem is not None else None
            agency = agency_elem.text if agency_elem is not None else None
            country = country_elem.text if country_elem is not None else None
            
            # Only create grant if we have some information
            if any([grant_id, acronym, agency, country]):
                return Grant(
                    grant_id=grant_id,
                    acronym=acronym,
                    agency=agency,
                    country=country
                )
            
        except Exception as e:
            logger.warning(f"Error extracting grant: {str(e)}")
        
        return None
    
    def _extract_pages(self, article_elem: ET.Element) -> Optional[str]:
        """Extract page numbers."""
        pagination_elem = article_elem.find('.//Pagination')
        if pagination_elem is not None:
            medline_pgn = pagination_elem.find('MedlinePgn')
            if medline_pgn is not None and medline_pgn.text:
                return medline_pgn.text.strip()
        return None
    
    def _extract_elocation_id(self, article_elem: ET.Element) -> Optional[str]:
        """Extract electronic location ID."""
        elocation_elem = article_elem.find('.//ELocationID')
        if elocation_elem is not None and elocation_elem.text:
            return elocation_elem.text.strip()
        return None
    
    def _extract_epub_date(self, article_elem: ET.Element) -> Optional[datetime]:
        """Extract electronic publication date."""
        epub_elem = article_elem.find('.//ArticleDate[@DateType="Electronic"]')
        if epub_elem is not None:
            return self._parse_date_element(epub_elem)
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common HTML entities
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        return text
    
    def validate_paper_metadata(self, paper: PaperMetadata) -> bool:
        """Validate that paper metadata meets minimum requirements.
        
        Args:
            paper: Paper metadata to validate
            
        Returns:
            bool: True if paper meets minimum requirements
        """
        # Must have PMID
        if not paper.pmid:
            return False
        
        # Must have title
        if not paper.title.strip():
            return False
        
        # Should have at least one author (warning, not failure)
        if not paper.authors:
            logger.warning(f"Paper {paper.pmid} has no authors")
        
        return True