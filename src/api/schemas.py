"""Pydantic models for API request/response validation and serialization."""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import re

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import constr, confloat, conint


# Base response models
class BaseResponse(BaseModel):
    """Base response model."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response model."""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HealthResponse(BaseResponse):
    """Health check response model."""
    status: str = Field(..., description="Overall health status")
    components: Dict[str, str] = Field(..., description="Component health status")
    version: str = Field(..., description="API version")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


# Prediction models
class PredictionRequest(BaseModel):
    """Base prediction request model."""
    features: Dict[str, Union[float, int, str]] = Field(..., description="Input features")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    model_version: Optional[str] = Field(None, description="Specific model version")
    return_probabilities: bool = Field(False, description="Whether to return class probabilities")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('features')
    def validate_features(cls, v):
        """Validate features dictionary."""
        if not v:
            raise ValueError("Features cannot be empty")
        
        # Check for reasonable number of features
        if len(v) > 1000:
            raise ValueError("Too many features (max 1000)")
        
        return v


class PredictionResponse(BaseResponse):
    """Prediction response model."""
    prediction: Union[int, float, str] = Field(..., description="Model prediction")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    model_name: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    requests: List[PredictionRequest] = Field(..., description="List of prediction requests")
    model_name: Optional[str] = Field(None, description="Default model for all requests")
    model_version: Optional[str] = Field(None, description="Default model version")
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    
    @validator('requests')
    def validate_requests(cls, v):
        """Validate batch requests."""
        if not v:
            raise ValueError("Requests list cannot be empty")
        
        if len(v) > 100:
            raise ValueError("Too many requests in batch (max 100)")
        
        return v


class BatchPredictionResponse(BaseResponse):
    """Batch prediction response model."""
    responses: List[PredictionResponse] = Field(..., description="Individual prediction responses")
    total_requests: int = Field(..., description="Total number of requests")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    average_processing_time_ms: float = Field(..., description="Average processing time per request")


# Drug-Gene Interaction models
class DrugGeneInteractionRequest(BaseModel):
    """Drug-gene interaction prediction request."""
    drug: constr(min_length=1, max_length=100) = Field(..., description="Drug name or identifier")
    gene: constr(min_length=1, max_length=50) = Field(..., description="Gene symbol or identifier")
    population: Optional[str] = Field(None, description="Population/ethnicity")
    additional_context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    @validator('drug')
    def validate_drug(cls, v):
        """Validate drug name."""
        # Remove special characters except hyphens and spaces
        if not re.match(r'^[a-zA-Z0-9\s\-]+$', v):
            raise ValueError("Drug name contains invalid characters")
        return v.strip()
    
    @validator('gene')
    def validate_gene(cls, v):
        """Validate gene symbol."""
        # Gene symbols are typically alphanumeric with some special characters
        if not re.match(r'^[A-Z0-9\*\-]+$', v.upper()):
            raise ValueError("Invalid gene symbol format")
        return v.upper().strip()


class InteractionStrength(str, Enum):
    """Interaction strength levels."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    UNKNOWN = "unknown"


class ClinicalSignificance(str, Enum):
    """Clinical significance levels."""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"


class DrugGeneInteractionResponse(BaseResponse):
    """Drug-gene interaction prediction response."""
    drug: str = Field(..., description="Drug name")
    gene: str = Field(..., description="Gene symbol")
    interaction_exists: bool = Field(..., description="Whether interaction exists")
    interaction_strength: InteractionStrength = Field(..., description="Strength of interaction")
    clinical_significance: ClinicalSignificance = Field(..., description="Clinical significance")
    confidence_score: confloat(ge=0.0, le=1.0) = Field(..., description="Prediction confidence")
    evidence_level: str = Field(..., description="Level of supporting evidence")
    mechanism: Optional[str] = Field(None, description="Interaction mechanism")
    recommendations: List[str] = Field(default_factory=list, description="Clinical recommendations")
    references: List[str] = Field(default_factory=list, description="Supporting references")


class BulkDrugGeneRequest(BaseModel):
    """Bulk drug-gene interaction analysis request."""
    interactions: List[DrugGeneInteractionRequest] = Field(..., description="List of drug-gene pairs")
    
    @validator('interactions')
    def validate_interactions(cls, v):
        """Validate interaction list."""
        if not v:
            raise ValueError("Interactions list cannot be empty")
        
        if len(v) > 50:
            raise ValueError("Too many interactions in bulk request (max 50)")
        
        return v


class BulkDrugGeneResponse(BaseResponse):
    """Bulk drug-gene interaction analysis response."""
    results: List[DrugGeneInteractionResponse] = Field(..., description="Interaction analysis results")
    total_interactions: int = Field(..., description="Total number of interactions analyzed")
    processing_time_ms: float = Field(..., description="Total processing time")


# Gene-Protein-Drug Interaction Graph models
class GraphNodeType(str, Enum):
    """Node type in the interaction graph."""
    GENE = "gene"
    PROTEIN = "protein"
    DRUG = "drug"


class GraphEdgeType(str, Enum):
    """Edge relationship type in the interaction graph."""
    ENCODES = "encodes"
    PGX_INTERACTION = "pgx_interaction"


class GraphNode(BaseModel):
    """A single node (gene, protein, or drug) in the interaction graph."""
    id: str = Field(..., description="Stable identifier (Ensembl gene ID, UniProt ID, or ChEMBL ID)")
    type: GraphNodeType = Field(..., description="Node type")
    label: str = Field(..., description="Display name")
    subtitle: Optional[str] = Field(None, description="Secondary display text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional node attributes")


class GraphEdge(BaseModel):
    """A single edge (relationship) between two nodes in the interaction graph."""
    id: str = Field(..., description="Deterministic edge identifier")
    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")
    relationship: GraphEdgeType = Field(..., description="Type of relationship")
    action_type: Optional[str] = Field(None, description="Raw pharmacogenomic category (dosage/toxicity/metabolism-pk/efficacy)")
    phenotype: Optional[str] = Field(None, description="Associated phenotype text")
    evidence_level: Optional[str] = Field(None, description="CPIC evidence level (e.g. 1A, 2B, 3)")
    annotation_count: int = Field(1, description="Number of raw source annotations aggregated into this edge")
    confidence: confloat(ge=0.0, le=1.0) = Field(0.3, description="Confidence score derived from evidence level")
    literature: List[str] = Field(default_factory=list, description="Supporting PubMed IDs (deduplicated across aggregated rows)")
    pharmgkb_ids: List[str] = Field(default_factory=list, description="PharmGKB (ClinPGx) clinical-annotation IDs backing this interaction")


class GraphSearchCandidate(BaseModel):
    """A single search result candidate for graph exploration."""
    id: str = Field(..., description="Entity identifier")
    entity_type: str = Field(..., description="Open Targets entity type: target or drug")
    name: str = Field(..., description="Display name")
    description: Optional[str] = Field(None, description="Short description")
    score: float = Field(..., description="Search relevance score")


class GraphSearchResponse(BaseResponse):
    """Response for a gene/protein/drug search query."""
    query: str = Field(..., description="Original search query")
    candidates: List[GraphSearchCandidate] = Field(default_factory=list, description="Matching entities")


class GraphExpandResponse(BaseResponse):
    """Response for expanding a node into its interaction subgraph."""
    center_node_id: str = Field(..., description="Id of the node that was expanded")
    nodes: List[GraphNode] = Field(default_factory=list, description="Nodes in the returned subgraph")
    edges: List[GraphEdge] = Field(default_factory=list, description="Edges in the returned subgraph")
    truncated: bool = Field(False, description="Whether results were capped by the limit")
    total_available: Optional[int] = Field(None, description="Total interactions available before truncation")


# Risk Scoring models
class GeneticVariant(BaseModel):
    """Genetic variant information."""
    gene: str = Field(..., description="Gene symbol")
    variant: str = Field(..., description="Variant identifier (e.g., rs number)")
    genotype: str = Field(..., description="Patient genotype")
    allele_frequency: Optional[float] = Field(None, description="Population allele frequency")
    
    @validator('genotype')
    def validate_genotype(cls, v):
        """Validate genotype format."""
        # Common genotype patterns: AA, AB, BB, *1/*2, etc.
        if not re.match(r'^[A-Z0-9\*\/]+$', v.upper()):
            raise ValueError("Invalid genotype format")
        return v.upper()


class PatientProfile(BaseModel):
    """Patient profile for risk assessment."""
    patient_id: constr(min_length=1, max_length=50) = Field(..., description="Patient identifier")
    age: Optional[conint(ge=0, le=150)] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    ethnicity: Optional[str] = Field(None, description="Patient ethnicity")
    weight_kg: Optional[confloat(gt=0, le=500)] = Field(None, description="Patient weight in kg")
    height_cm: Optional[confloat(gt=0, le=300)] = Field(None, description="Patient height in cm")
    genetic_variants: List[GeneticVariant] = Field(..., description="Patient genetic variants")
    medical_history: Optional[List[str]] = Field(None, description="Relevant medical history")
    current_medications: Optional[List[str]] = Field(None, description="Current medications")
    
    @validator('genetic_variants')
    def validate_variants(cls, v):
        """Validate genetic variants."""
        if not v:
            raise ValueError("At least one genetic variant is required")
        
        if len(v) > 100:
            raise ValueError("Too many genetic variants (max 100)")
        
        return v


class RiskLevel(str, Enum):
    """Risk level categories."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class RiskScoringRequest(BaseModel):
    """Risk scoring request."""
    patient: PatientProfile = Field(..., description="Patient profile")
    drug: str = Field(..., description="Drug for risk assessment")
    indication: Optional[str] = Field(None, description="Medical indication")
    dose: Optional[str] = Field(None, description="Proposed dose")
    
    @validator('drug')
    def validate_drug(cls, v):
        """Validate drug name."""
        if not re.match(r'^[a-zA-Z0-9\s\-]+$', v):
            raise ValueError("Drug name contains invalid characters")
        return v.strip()


class RiskFactor(BaseModel):
    """Individual risk factor."""
    factor: str = Field(..., description="Risk factor name")
    contribution: confloat(ge=0.0, le=1.0) = Field(..., description="Contribution to overall risk")
    description: str = Field(..., description="Risk factor description")
    evidence_level: str = Field(..., description="Evidence level")


class RiskScoringResponse(BaseResponse):
    """Risk scoring response."""
    patient_id: str = Field(..., description="Patient identifier")
    drug: str = Field(..., description="Drug assessed")
    overall_risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_score: confloat(ge=0.0, le=1.0) = Field(..., description="Numerical risk score")
    risk_factors: List[RiskFactor] = Field(..., description="Contributing risk factors")
    population_comparison: Optional[str] = Field(None, description="Comparison to population average")
    recommendations: List[str] = Field(default_factory=list, description="Risk mitigation recommendations")
    monitoring_suggestions: List[str] = Field(default_factory=list, description="Monitoring suggestions")
    contraindications: List[str] = Field(default_factory=list, description="Contraindications")


# Dosing models
class DosingRequest(BaseModel):
    """Personalized dosing request."""
    patient: PatientProfile = Field(..., description="Patient profile")
    drug: str = Field(..., description="Drug for dosing")
    indication: str = Field(..., description="Medical indication")
    target_efficacy: Optional[confloat(ge=0.0, le=1.0)] = Field(0.8, description="Target efficacy level")
    risk_tolerance: Optional[confloat(ge=0.0, le=1.0)] = Field(0.1, description="Risk tolerance level")
    
    @validator('indication')
    def validate_indication(cls, v):
        """Validate medical indication."""
        if len(v.strip()) < 3:
            raise ValueError("Medical indication too short")
        return v.strip()


class DoseRecommendation(BaseModel):
    """Dose recommendation."""
    dose_amount: str = Field(..., description="Recommended dose amount")
    dose_frequency: str = Field(..., description="Dosing frequency")
    route: str = Field(..., description="Route of administration")
    duration: Optional[str] = Field(None, description="Treatment duration")
    adjustments: List[str] = Field(default_factory=list, description="Dose adjustments")
    rationale: str = Field(..., description="Rationale for recommendation")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence in recommendation")


class DosingResponse(BaseResponse):
    """Personalized dosing response."""
    patient_id: str = Field(..., description="Patient identifier")
    drug: str = Field(..., description="Drug")
    indication: str = Field(..., description="Medical indication")
    primary_recommendation: DoseRecommendation = Field(..., description="Primary dose recommendation")
    alternative_recommendations: List[DoseRecommendation] = Field(default_factory=list, description="Alternative options")
    safety_considerations: List[str] = Field(default_factory=list, description="Safety considerations")
    monitoring_parameters: List[str] = Field(default_factory=list, description="Parameters to monitor")
    drug_interactions: List[str] = Field(default_factory=list, description="Potential drug interactions")
    contraindications: List[str] = Field(default_factory=list, description="Contraindications")


# Clinical Decision Support models
class ClinicalContext(BaseModel):
    """Clinical context for decision support."""
    patient: PatientProfile = Field(..., description="Patient profile")
    clinical_scenario: str = Field(..., description="Clinical scenario description")
    treatment_goals: List[str] = Field(..., description="Treatment goals")
    constraints: Optional[List[str]] = Field(None, description="Treatment constraints")
    urgency: Optional[str] = Field("routine", description="Urgency level")
    
    @validator('clinical_scenario')
    def validate_scenario(cls, v):
        """Validate clinical scenario."""
        if len(v.strip()) < 10:
            raise ValueError("Clinical scenario description too short")
        return v.strip()


class ClinicalRecommendation(BaseModel):
    """Clinical recommendation."""
    recommendation_type: str = Field(..., description="Type of recommendation")
    recommendation: str = Field(..., description="Recommendation text")
    strength: str = Field(..., description="Recommendation strength")
    evidence_level: str = Field(..., description="Evidence level")
    rationale: str = Field(..., description="Rationale")
    alternatives: List[str] = Field(default_factory=list, description="Alternative options")
    monitoring: List[str] = Field(default_factory=list, description="Monitoring requirements")


class ClinicalSupportRequest(BaseModel):
    """Clinical decision support request."""
    context: ClinicalContext = Field(..., description="Clinical context")
    requested_support: List[str] = Field(..., description="Types of support requested")
    
    @validator('requested_support')
    def validate_support_types(cls, v):
        """Validate requested support types."""
        valid_types = [
            "drug_selection", "dosing", "risk_assessment", 
            "monitoring", "drug_interactions", "contraindications"
        ]
        
        for support_type in v:
            if support_type not in valid_types:
                raise ValueError(f"Invalid support type: {support_type}")
        
        return v


class ClinicalSupportResponse(BaseResponse):
    """Clinical decision support response."""
    patient_id: str = Field(..., description="Patient identifier")
    clinical_scenario: str = Field(..., description="Clinical scenario")
    recommendations: List[ClinicalRecommendation] = Field(..., description="Clinical recommendations")
    risk_assessment: Optional[RiskScoringResponse] = Field(None, description="Risk assessment")
    dosing_guidance: Optional[DosingResponse] = Field(None, description="Dosing guidance")
    drug_interactions: List[str] = Field(default_factory=list, description="Drug interactions")
    alerts: List[str] = Field(default_factory=list, description="Clinical alerts")
    references: List[str] = Field(default_factory=list, description="Supporting references")
    confidence_score: confloat(ge=0.0, le=1.0) = Field(..., description="Overall confidence")


# Authentication models
class LoginRequest(BaseModel):
    """Login request."""
    username: constr(min_length=3, max_length=50) = Field(..., description="Username")
    password: constr(min_length=8, max_length=100) = Field(..., description="Password")


class TokenResponse(BaseResponse):
    """Token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")


class UserInfo(BaseModel):
    """User information."""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="Email address")
    roles: List[str] = Field(..., description="User roles")
    permissions: List[str] = Field(..., description="User permissions")


# Admin models
class ModelInfo(BaseModel):
    """Model information."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Model stage")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    created_at: datetime = Field(..., description="Creation timestamp")
    description: Optional[str] = Field(None, description="Model description")


class SystemMetrics(BaseModel):
    """System metrics."""
    total_requests: int = Field(..., description="Total API requests")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    average_response_time_ms: float = Field(..., description="Average response time")
    active_models: int = Field(..., description="Number of active models")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    uptime_seconds: float = Field(..., description="System uptime")


# Validation utilities
def validate_patient_id(patient_id: str) -> str:
    """Validate patient ID format."""
    if not re.match(r'^[A-Za-z0-9\-_]+$', patient_id):
        raise ValueError("Patient ID contains invalid characters")
    return patient_id


def validate_drug_name(drug_name: str) -> str:
    """Validate drug name format."""
    if not re.match(r'^[a-zA-Z0-9\s\-]+$', drug_name):
        raise ValueError("Drug name contains invalid characters")
    return drug_name.strip()


def validate_gene_symbol(gene_symbol: str) -> str:
    """Validate gene symbol format."""
    if not re.match(r'^[A-Z0-9\*\-]+$', gene_symbol.upper()):
        raise ValueError("Invalid gene symbol format")
    return gene_symbol.upper().strip()