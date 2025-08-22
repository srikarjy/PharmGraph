"""Data models for structured paper information from NCBI."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class PublicationType(Enum):
    """Publication types from PubMed."""
    JOURNAL_ARTICLE = "Journal Article"
    REVIEW = "Review"
    CLINICAL_TRIAL = "Clinical Trial"
    META_ANALYSIS = "Meta-Analysis"
    SYSTEMATIC_REVIEW = "Systematic Review"
    CASE_REPORT = "Case Reports"
    EDITORIAL = "Editorial"
    LETTER = "Letter"
    COMMENT = "Comment"
    RETRACTED_PUBLICATION = "Retracted Publication"
    RANDOMIZED_CONTROLLED_TRIAL = "Randomized Controlled Trial"
    MULTICENTER_STUDY = "Multicenter Study"
    COMPARATIVE_STUDY = "Comparative Study"
    CLINICAL_TRIAL_PHASE_I = "Clinical Trial, Phase I"
    CLINICAL_TRIAL_PHASE_II = "Clinical Trial, Phase II"
    CLINICAL_TRIAL_PHASE_III = "Clinical Trial, Phase III"
    CLINICAL_TRIAL_PHASE_IV = "Clinical Trial, Phase IV"


@dataclass
class Author:
    """Author information from PubMed."""
    last_name: str
    first_name: Optional[str] = None
    initials: Optional[str] = None
    suffix: Optional[str] = None
    affiliation: Optional[str] = None
    orcid: Optional[str] = None
    
    @property
    def full_name(self) -> str:
        """Get full name of author."""
        parts = []
        if self.first_name:
            parts.append(self.first_name)
        if self.last_name:
            parts.append(self.last_name)
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(parts)
    
    @property
    def citation_name(self) -> str:
        """Get name in citation format (Last, F.)."""
        if self.last_name and self.initials:
            return f"{self.last_name}, {self.initials}"
        elif self.last_name and self.first_name:
            first_initial = self.first_name[0].upper() if self.first_name else ""
            return f"{self.last_name}, {first_initial}."
        else:
            return self.last_name or "Unknown"


@dataclass
class MeshTerm:
    """MeSH (Medical Subject Headings) term information."""
    descriptor_name: str
    descriptor_ui: Optional[str] = None  # Unique identifier
    qualifier_name: Optional[str] = None
    qualifier_ui: Optional[str] = None
    major_topic: bool = False  # Whether this is a major topic
    
    def __str__(self) -> str:
        """String representation of MeSH term."""
        if self.qualifier_name:
            return f"{self.descriptor_name}/{self.qualifier_name}"
        return self.descriptor_name


@dataclass
class Grant:
    """Grant/funding information."""
    grant_id: Optional[str] = None
    acronym: Optional[str] = None
    agency: Optional[str] = None
    country: Optional[str] = None


@dataclass
class Journal:
    """Journal information."""
    title: str
    iso_abbreviation: Optional[str] = None
    issn: Optional[str] = None
    issn_type: Optional[str] = None  # Print, Electronic
    volume: Optional[str] = None
    issue: Optional[str] = None
    pub_date: Optional[datetime] = None
    
    @property
    def citation_format(self) -> str:
        """Get journal in citation format."""
        parts = [self.iso_abbreviation or self.title]
        if self.pub_date:
            parts.append(str(self.pub_date.year))
        if self.volume:
            parts.append(f"Vol. {self.volume}")
        if self.issue:
            parts.append(f"No. {self.issue}")
        return ". ".join(parts)


@dataclass
class AbstractSection:
    """Section of a structured abstract."""
    label: Optional[str] = None  # BACKGROUND, METHODS, RESULTS, CONCLUSIONS
    text: str = ""
    
    def __str__(self) -> str:
        """String representation of abstract section."""
        if self.label:
            return f"{self.label}: {self.text}"
        return self.text


@dataclass
class PaperMetadata:
    """Complete metadata for a research paper from PubMed."""
    
    # Core identifiers
    pmid: str
    pmc_id: Optional[str] = None
    doi: Optional[str] = None
    
    # Basic information
    title: str = ""
    abstract: str = ""
    abstract_sections: List[AbstractSection] = field(default_factory=list)
    
    # Authors and affiliations
    authors: List[Author] = field(default_factory=list)
    
    # Journal information
    journal: Optional[Journal] = None
    
    # Publication details
    publication_date: Optional[datetime] = None
    epub_date: Optional[datetime] = None
    received_date: Optional[datetime] = None
    accepted_date: Optional[datetime] = None
    revised_date: Optional[datetime] = None
    
    # Classification and indexing
    mesh_terms: List[MeshTerm] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)
    
    # Language and location
    language: str = "eng"
    country: Optional[str] = None
    
    # Funding information
    grants: List[Grant] = field(default_factory=list)
    
    # Additional metadata
    pages: Optional[str] = None
    elocation_id: Optional[str] = None  # Electronic location ID
    
    # Processing metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    source_xml: Optional[str] = None  # Original XML for debugging
    
    # Computed properties
    @property
    def first_author(self) -> Optional[Author]:
        """Get first author."""
        return self.authors[0] if self.authors else None
    
    @property
    def last_author(self) -> Optional[Author]:
        """Get last author."""
        return self.authors[-1] if self.authors else None
    
    @property
    def author_count(self) -> int:
        """Get number of authors."""
        return len(self.authors)
    
    @property
    def publication_year(self) -> Optional[int]:
        """Get publication year."""
        return self.publication_date.year if self.publication_date else None
    
    @property
    def has_abstract(self) -> bool:
        """Check if paper has abstract."""
        return bool(self.abstract.strip())
    
    @property
    def is_clinical_trial(self) -> bool:
        """Check if paper is a clinical trial."""
        clinical_trial_types = [
            "Clinical Trial",
            "Randomized Controlled Trial",
            "Clinical Trial, Phase I",
            "Clinical Trial, Phase II", 
            "Clinical Trial, Phase III",
            "Clinical Trial, Phase IV"
        ]
        return any(pt in clinical_trial_types for pt in self.publication_types)
    
    @property
    def is_review(self) -> bool:
        """Check if paper is a review."""
        review_types = ["Review", "Systematic Review", "Meta-Analysis"]
        return any(pt in review_types for pt in self.publication_types)
    
    @property
    def mesh_descriptors(self) -> List[str]:
        """Get list of MeSH descriptor names."""
        return [term.descriptor_name for term in self.mesh_terms]
    
    @property
    def major_mesh_terms(self) -> List[MeshTerm]:
        """Get MeSH terms marked as major topics."""
        return [term for term in self.mesh_terms if term.major_topic]
    
    def get_abstract_section(self, label: str) -> Optional[str]:
        """Get text from specific abstract section.
        
        Args:
            label: Section label (e.g., 'BACKGROUND', 'METHODS')
            
        Returns:
            Optional[str]: Section text if found
        """
        for section in self.abstract_sections:
            if section.label and section.label.upper() == label.upper():
                return section.text
        return None
    
    def has_mesh_term(self, descriptor: str) -> bool:
        """Check if paper has specific MeSH term.
        
        Args:
            descriptor: MeSH descriptor name
            
        Returns:
            bool: True if MeSH term is present
        """
        return descriptor.lower() in [term.descriptor_name.lower() for term in self.mesh_terms]
    
    def has_keyword(self, keyword: str) -> bool:
        """Check if paper has specific keyword.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            bool: True if keyword is present
        """
        return keyword.lower() in [kw.lower() for kw in self.keywords]
    
    def get_citation(self, style: str = "apa") -> str:
        """Generate citation for the paper.
        
        Args:
            style: Citation style ('apa', 'mla', 'chicago')
            
        Returns:
            str: Formatted citation
        """
        if style.lower() == "apa":
            return self._get_apa_citation()
        elif style.lower() == "mla":
            return self._get_mla_citation()
        else:
            return self._get_basic_citation()
    
    def _get_apa_citation(self) -> str:
        """Generate APA style citation."""
        parts = []
        
        # Authors
        if self.authors:
            if len(self.authors) == 1:
                parts.append(self.authors[0].citation_name)
            elif len(self.authors) <= 7:
                author_names = [author.citation_name for author in self.authors]
                parts.append(", ".join(author_names[:-1]) + ", & " + author_names[-1])
            else:
                author_names = [author.citation_name for author in self.authors[:6]]
                parts.append(", ".join(author_names) + ", ... " + self.authors[-1].citation_name)
        
        # Year
        if self.publication_year:
            parts.append(f"({self.publication_year})")
        
        # Title
        if self.title:
            parts.append(f"{self.title}.")
        
        # Journal
        if self.journal:
            parts.append(f"*{self.journal.title}*")
            if self.journal.volume:
                volume_part = f"*{self.journal.volume}*"
                if self.journal.issue:
                    volume_part += f"({self.journal.issue})"
                parts.append(volume_part)
        
        # Pages or DOI
        if self.pages:
            parts.append(self.pages)
        elif self.doi:
            parts.append(f"https://doi.org/{self.doi}")
        
        return " ".join(parts)
    
    def _get_basic_citation(self) -> str:
        """Generate basic citation."""
        parts = []
        
        if self.authors:
            if len(self.authors) == 1:
                parts.append(self.authors[0].full_name)
            else:
                parts.append(f"{self.authors[0].full_name} et al.")
        
        if self.title:
            parts.append(f'"{self.title}"')
        
        if self.journal:
            parts.append(self.journal.title)
        
        if self.publication_year:
            parts.append(f"({self.publication_year})")
        
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pmid": self.pmid,
            "pmc_id": self.pmc_id,
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
            "authors": [
                {
                    "last_name": author.last_name,
                    "first_name": author.first_name,
                    "initials": author.initials,
                    "affiliation": author.affiliation,
                    "orcid": author.orcid
                }
                for author in self.authors
            ],
            "journal": {
                "title": self.journal.title,
                "iso_abbreviation": self.journal.iso_abbreviation,
                "issn": self.journal.issn,
                "volume": self.journal.volume,
                "issue": self.journal.issue
            } if self.journal else None,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "mesh_terms": [
                {
                    "descriptor_name": term.descriptor_name,
                    "qualifier_name": term.qualifier_name,
                    "major_topic": term.major_topic
                }
                for term in self.mesh_terms
            ],
            "keywords": self.keywords,
            "publication_types": self.publication_types,
            "language": self.language,
            "grants": [
                {
                    "grant_id": grant.grant_id,
                    "agency": grant.agency,
                    "country": grant.country
                }
                for grant in self.grants
            ],
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperMetadata':
        """Create instance from dictionary."""
        # This is a simplified implementation
        # In practice, you'd want more robust deserialization
        paper = cls(
            pmid=data["pmid"],
            title=data.get("title", ""),
            abstract=data.get("abstract", "")
        )
        
        # Add authors
        for author_data in data.get("authors", []):
            author = Author(
                last_name=author_data["last_name"],
                first_name=author_data.get("first_name"),
                initials=author_data.get("initials"),
                affiliation=author_data.get("affiliation"),
                orcid=author_data.get("orcid")
            )
            paper.authors.append(author)
        
        # Add other fields as needed
        paper.keywords = data.get("keywords", [])
        paper.publication_types = data.get("publication_types", [])
        paper.language = data.get("language", "eng")
        
        return paper