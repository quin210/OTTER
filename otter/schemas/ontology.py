"""Four-layer ontology contracts: concept graph, rule hypergraph,
pedagogical graph and provenance links (see spec section 3.2)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Subject = Literal["physics", "chemistry", "biology", "history", "geography", "literature", "other"]
ConceptType = Literal[
    "Concept", "Process", "Quantity", "Unit", "Law", "Event", "Person", "Place", "Period",
    "State", "Entity", "Organism", "Structure", "Substance", "Phenomenon", "Work", "LiteraryDevice", "Other",
]
RelationType = Literal[
    "is_a", "part_of", "has_part", "has_property", "involves", "affects", "causes", "caused_by",
    "occurs_before", "occurs_after", "occurs_in", "located_in", "measured_in", "defined_as",
    "example_of", "contrasts_with", "synonym_of", "authored_by", "belongs_to_period", "related_to",
]
KnowledgeStatus = Literal["explicit_fact", "explicit_rule", "derived_conclusion", "hypothesized_relation"]
SupportStatus = Literal["supported", "partially_supported", "unsupported", "needs_review"]


class Validation(BaseModel):
    schema_valid: bool = True
    source_ids_exist: bool = False
    source_support: SupportStatus = "needs_review"   # set by validator, never by the extractor
    support_score: Optional[float] = None
    human_verified: Optional[bool] = None
    notes: list[str] = Field(default_factory=list)


class Concept(BaseModel):
    concept_id: str                                  # "<subject>:c:<slug>" ; same term in two subjects => two ids
    label: str                                       # Vietnamese canonical label
    aliases: list[str] = Field(default_factory=list)
    concept_type: ConceptType = "Concept"
    subject: Subject
    grade: Optional[int] = None
    definition: Optional[str] = None
    source_span_ids: list[str] = Field(default_factory=list)
    validation: Validation = Field(default_factory=Validation)


class SemanticRelation(BaseModel):
    relation_id: str
    subject_id: str                                  # concept_id
    relation_type: RelationType
    object_id: str                                   # concept_id
    status: KnowledgeStatus = "explicit_fact"
    qualifier: Optional[str] = None                  # scope / condition text
    source_span_ids: list[str] = Field(default_factory=list)
    validation: Validation = Field(default_factory=Validation)


class Fact(BaseModel):
    """Atomic textbook statement (assertion) not reducible to a binary relation."""
    assertion_id: str
    text: str                                        # normalized Vietnamese statement
    subject: Subject
    concept_ids: list[str] = Field(default_factory=list)
    status: KnowledgeStatus = "explicit_fact"
    source_span_ids: list[str] = Field(default_factory=list)
    validation: Validation = Field(default_factory=Validation)


class Atom(BaseModel):
    """Grounded proposition used in rules: predicate(args). Kept simple on purpose."""
    predicate: str                                   # snake_case, subject-scoped, e.g. "net_force_zero"
    args: list[str] = Field(default_factory=list)    # concept ids or variables ("?x")
    text: str = ""                                   # Vietnamese surface form
    negated: bool = False


class ConditionalRule(BaseModel):
    rule_id: str
    subject: Subject
    rule_type: Literal["conditional", "definitional", "causal", "formula", "classification", "temporal"] = "conditional"
    premises: list[Atom]
    guards: list[Atom] = Field(default_factory=list)       # applicability conditions; all must be entailed
    conclusions: list[Atom]
    exceptions: list[Atom] = Field(default_factory=list)
    formula: Optional[str] = None                          # e.g. "a = F / m" for rule_type=formula
    variables: dict[str, str] = Field(default_factory=dict)  # symbol -> quantity concept id / unit
    learning_objective_ids: list[str] = Field(default_factory=list)
    status: KnowledgeStatus = "explicit_rule"
    source_span_ids: list[str] = Field(default_factory=list)
    validation: Validation = Field(default_factory=Validation)


class LearningObjective(BaseModel):
    objective_id: str
    text: str
    lesson_section_id: Optional[str] = None
    subject: Subject
    grade: Optional[int] = None
    source_span_ids: list[str] = Field(default_factory=list)


class PedagogicalLink(BaseModel):
    link_id: str
    source_id: str                                   # concept/rule/objective/section id
    link_type: Literal["teaches", "assesses", "requires_learning", "exemplifies"]
    target_id: str
    source_span_ids: list[str] = Field(default_factory=list)


class TextbookKnowledgeUnit(BaseModel):
    """Builder output bundle per source span group (one extraction call)."""
    tku_id: str
    book_id: str
    section_id: Optional[str] = None
    span_ids: list[str]
    concepts: list[Concept] = Field(default_factory=list)
    relations: list[SemanticRelation] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    rules: list[ConditionalRule] = Field(default_factory=list)
    objectives: list[LearningObjective] = Field(default_factory=list)
    pedagogical_links: list[PedagogicalLink] = Field(default_factory=list)
    extractor: str = ""                              # model id + prompt version
    content_hash: str = ""                           # cache key over span texts + extractor
