"""
Multi-Agent Intelligence System für Bundeskanzler-KI
===================================================

Architecture: Coordinator + Specialized Agent System
- Master Coordinator Agent für Query Routing & Response Synthesis
- Politik Agent: Bundestag, Gesetze, Wahlen, Koalitionen
- Wirtschaft Agent: Finanzpolitik, Arbeitsmarkt, Inflation, EU
- Klima Agent: Energiewende, CO2-Ziele, Verkehrswende, Klimaschutz

Author: Advanced RAG Team
Date: September 15, 2025
Version: 1.0
"""

import asyncio
import json
import logging
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

# Import our Advanced RAG System as foundation
try:
    from .advanced_rag_system import AdvancedRAGSystem
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from core.advanced_rag_system import AdvancedRAGSystem

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Enum for different agent types in the system"""
    COORDINATOR = "coordinator"
    POLITIK = "politik"
    WIRTSCHAFT = "wirtschaft" 
    KLIMA = "klima"

class QueryComplexity(Enum):
    """Classification of query complexity levels"""
    SIMPLE = "simple"          # Single domain, direct answer
    MODERATE = "moderate"      # Cross-domain or analytical
    COMPLEX = "complex"        # Multi-domain synthesis required

@dataclass
class AgentCapability:
    """Defines capabilities and expertise areas of an agent"""
    domain: str
    expertise_areas: List[str]
    keywords: List[str]
    confidence_threshold: float = 0.7
    
@dataclass
class QueryClassification:
    """Result of query analysis and routing decision"""
    primary_domain: str
    secondary_domains: List[str]
    complexity: QueryComplexity
    keywords_found: List[str]
    confidence: float
    routing_decision: List[AgentType]

@dataclass
class AgentResponse:
    """Standardized response from any agent"""
    agent_type: AgentType
    content: str
    confidence: float
    sources: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class MultiAgentResponse:
    """Final synthesized response from multiple agents"""
    primary_response: str
    contributing_agents: List[AgentType]
    confidence: float
    sources: List[Dict[str, Any]]
    synthesis_metadata: Dict[str, Any]
    total_processing_time: float

class BaseAgent(ABC):
    """Abstract base class for all agents in the system"""
    
    def __init__(self, agent_type: AgentType, capability: AgentCapability, rag_system: AdvancedRAGSystem):
        self.agent_type = agent_type
        self.capability = capability
        self.rag_system = rag_system
        self.stats = {
            'queries_processed': 0,
            'avg_response_time': 0.0,
            'avg_confidence': 0.0,
            'successful_responses': 0
        }
        
    @abstractmethod
    async def process_query(self, query: str, context: Optional[Dict] = None) -> AgentResponse:
        """Process a query and return agent-specific response"""
        pass
        
    def calculate_query_relevance(self, query: str) -> float:
        """Calculate how relevant this query is to this agent's expertise"""
        query_lower = query.lower()
        
        # Check for domain keywords
        keyword_matches = sum(1 for keyword in self.capability.keywords 
                            if keyword.lower() in query_lower)
        keyword_score = min(keyword_matches / max(len(self.capability.keywords), 1), 1.0)
        
        # Check for expertise area mentions
        expertise_matches = sum(1 for area in self.capability.expertise_areas
                              if area.lower() in query_lower)
        expertise_score = min(expertise_matches / max(len(self.capability.expertise_areas), 1), 1.0)
        
        # Combined relevance score
        relevance = (keyword_score * 0.6) + (expertise_score * 0.4)
        
        return relevance
        
    def update_stats(self, response_time: float, confidence: float, successful: bool):
        """Update agent performance statistics"""
        self.stats['queries_processed'] += 1
        
        # Running average for response time
        current_avg_time = self.stats['avg_response_time']
        n = self.stats['queries_processed']
        self.stats['avg_response_time'] = ((current_avg_time * (n-1)) + response_time) / n
        
        # Running average for confidence
        current_avg_conf = self.stats['avg_confidence']
        self.stats['avg_confidence'] = ((current_avg_conf * (n-1)) + confidence) / n
        
        if successful:
            self.stats['successful_responses'] += 1

class PolitikAgent(BaseAgent):
    """Specialized agent for Politik domain queries"""
    
    def __init__(self, rag_system: AdvancedRAGSystem):
        capability = AgentCapability(
            domain="Politik",
            expertise_areas=[
                "Bundestag", "Bundesrat", "Gesetze", "Gesetzgebung",
                "Wahlen", "Koalition", "Opposition", "Parteien",
                "Verfassung", "Grundgesetz", "Demokratie", "Regierung",
                "Kanzler", "Minister", "Parlament", "Abgeordnete"
            ],
            keywords=[
                "politik", "politisch", "bundestag", "bundesrat", "gesetz", "wahl",
                "koalition", "partei", "spd", "cdu", "grüne", "fdp", "linke", "afd",
                "kanzler", "minister", "regierung", "opposition", "demokratie",
                "parlament", "abgeordnete", "verfassung", "grundgesetz"
            ],
            confidence_threshold=0.75
        )
        super().__init__(AgentType.POLITIK, capability, rag_system)
        
    async def process_query(self, query: str, context: Optional[Dict] = None) -> AgentResponse:
        """Process politik-specific queries with enhanced context"""
        start_time = time.time()
        
        try:
            # Enhanced query for politik domain
            enhanced_query = self._enhance_politik_query(query)
            
            # Use RAG system to get relevant documents
            documents = self.rag_system.retrieve_relevant_documents(enhanced_query, top_k=8)
            
            # Filter for politik-relevant content
            politik_docs = self._filter_politik_documents(documents)
            
            # Generate politik-focused response
            response_content = self._generate_politik_response(query, politik_docs)
            
            processing_time = time.time() - start_time
            confidence = self._calculate_response_confidence(politik_docs)
            
            response = AgentResponse(
                agent_type=self.agent_type,
                content=response_content,
                confidence=confidence,
                sources=politik_docs[:5],  # Top 5 sources
                processing_time=processing_time,
                metadata={
                    'enhanced_query': enhanced_query,
                    'documents_found': len(documents),
                    'politik_documents': len(politik_docs),
                    'domain_expertise': self.capability.domain
                }
            )
            
            self.update_stats(processing_time, confidence, True)
            return response
            
        except Exception as e:
            logger.error(f"Politik Agent error: {str(e)}")
            processing_time = time.time() - start_time
            
            response = AgentResponse(
                agent_type=self.agent_type,
                content=f"Entschuldigung, ich konnte Ihre Politik-Frage nicht vollständig beantworten: {str(e)}",
                confidence=0.0,
                sources=[],
                processing_time=processing_time,
                metadata={'error': str(e)}
            )
            
            self.update_stats(processing_time, 0.0, False)
            return response
    
    def _enhance_politik_query(self, query: str) -> str:
        """Enhance query with politik-specific context"""
        politik_context = [
            "Bundesregierung", "Bundestag", "deutsche Politik",
            "Gesetzgebung", "politische Entscheidungen"
        ]
        
        # Add relevant context terms
        enhanced = f"{query} {' '.join(politik_context[:2])}"
        return enhanced
    
    def _filter_politik_documents(self, documents: List[Dict]) -> List[Dict]:
        """Filter documents for politik relevance"""
        politik_keywords = self.capability.keywords
        
        filtered_docs = []
        for doc in documents:
            text_lower = doc['text'].lower()
            
            # Count politik keyword matches
            matches = sum(1 for keyword in politik_keywords if keyword in text_lower)
            
            if matches >= 2:  # At least 2 politik keywords
                doc['politik_relevance'] = matches / len(politik_keywords)
                filtered_docs.append(doc)
        
        # Sort by politik relevance and original score
        filtered_docs.sort(key=lambda x: (x.get('politik_relevance', 0), x['score']), reverse=True)
        
        return filtered_docs
    
    def _generate_politik_response(self, query: str, documents: List[Dict]) -> str:
        """Generate politik-focused response"""
        if not documents:
            return "Ich konnte keine relevanten politischen Informationen zu Ihrer Frage finden."
        
        # Extract key information from top documents
        key_points = []
        for doc in documents[:3]:
            text = doc['text']
            # Extract sentences containing politik keywords
            sentences = text.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in self.capability.keywords[:10]):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                key_points.extend(relevant_sentences[:2])  # Top 2 sentences per doc
        
        if key_points:
            response = f"🏛️ **Politik-Analyse:**\n\n"
            for i, point in enumerate(key_points[:4], 1):
                if point:
                    response += f"{i}. {point}\n\n"
            
            response += f"📊 **Quelle:** Basiert auf {len(documents)} politischen Dokumenten"
            return response
        else:
            return f"📋 **Politik-Information:** {documents[0]['text']}"
    
    def _calculate_response_confidence(self, documents: List[Dict]) -> float:
        """Calculate confidence based on document quality and relevance"""
        if not documents:
            return 0.0
        
        # Average document scores
        avg_score = sum(doc['score'] for doc in documents) / len(documents)
        
        # Politik relevance bonus
        avg_relevance = sum(doc.get('politik_relevance', 0) for doc in documents) / len(documents)
        
        # Combined confidence
        confidence = (avg_score * 0.7) + (avg_relevance * 0.3)
        
        return min(confidence, 0.95)  # Cap at 95%

class WirtschaftAgent(BaseAgent):
    """Specialized agent for Wirtschaft domain queries"""
    
    def __init__(self, rag_system: AdvancedRAGSystem):
        capability = AgentCapability(
            domain="Wirtschaft",
            expertise_areas=[
                "Finanzpolitik", "Arbeitsmarkt", "Inflation", "Zinsen",
                "EU-Politik", "Handel", "Export", "Import", "BIP",
                "Steuern", "Haushalt", "Schulden", "Investitionen",
                "Unternehmen", "Mittelstand", "Startups", "Innovation"
            ],
            keywords=[
                "wirtschaft", "ökonomie", "finanzen", "geld", "euro", "inflation",
                "arbeitsmarkt", "arbeitslosigkeit", "job", "arbeit", "lohn", "gehalt",
                "steuer", "haushalt", "schulden", "investition", "bip", "wachstum",
                "unternehmen", "mittelstand", "handel", "export", "import", "eu",
                "zinsen", "ezb", "bundesbank", "börse", "aktie"
            ],
            confidence_threshold=0.75
        )
        super().__init__(AgentType.WIRTSCHAFT, capability, rag_system)
        
    async def process_query(self, query: str, context: Optional[Dict] = None) -> AgentResponse:
        """Process wirtschaft-specific queries"""
        start_time = time.time()
        
        try:
            # Enhanced query for wirtschaft domain
            enhanced_query = self._enhance_wirtschaft_query(query)
            
            # Use RAG system to get relevant documents
            documents = self.rag_system.retrieve_relevant_documents(enhanced_query, top_k=8)
            
            # Filter for wirtschaft-relevant content
            wirtschaft_docs = self._filter_wirtschaft_documents(documents)
            
            # Generate wirtschaft-focused response
            response_content = self._generate_wirtschaft_response(query, wirtschaft_docs)
            
            processing_time = time.time() - start_time
            confidence = self._calculate_response_confidence(wirtschaft_docs)
            
            response = AgentResponse(
                agent_type=self.agent_type,
                content=response_content,
                confidence=confidence,
                sources=wirtschaft_docs[:5],
                processing_time=processing_time,
                metadata={
                    'enhanced_query': enhanced_query,
                    'documents_found': len(documents),
                    'wirtschaft_documents': len(wirtschaft_docs),
                    'domain_expertise': self.capability.domain
                }
            )
            
            self.update_stats(processing_time, confidence, True)
            return response
            
        except Exception as e:
            logger.error(f"Wirtschaft Agent error: {str(e)}")
            processing_time = time.time() - start_time
            
            response = AgentResponse(
                agent_type=self.agent_type,
                content=f"Entschuldigung, ich konnte Ihre Wirtschafts-Frage nicht vollständig beantworten: {str(e)}",
                confidence=0.0,
                sources=[],
                processing_time=processing_time,
                metadata={'error': str(e)}
            )
            
            self.update_stats(processing_time, 0.0, False)
            return response
    
    def _enhance_wirtschaft_query(self, query: str) -> str:
        """Enhance query with wirtschaft-specific context"""
        wirtschaft_context = [
            "deutsche Wirtschaft", "Finanzpolitik", "Arbeitsmarkt",
            "wirtschaftliche Entwicklung", "ökonomische Maßnahmen"
        ]
        
        enhanced = f"{query} {' '.join(wirtschaft_context[:2])}"
        return enhanced
    
    def _filter_wirtschaft_documents(self, documents: List[Dict]) -> List[Dict]:
        """Filter documents for wirtschaft relevance"""
        wirtschaft_keywords = self.capability.keywords
        
        filtered_docs = []
        for doc in documents:
            text_lower = doc['text'].lower()
            
            matches = sum(1 for keyword in wirtschaft_keywords if keyword in text_lower)
            
            if matches >= 2:
                doc['wirtschaft_relevance'] = matches / len(wirtschaft_keywords)
                filtered_docs.append(doc)
        
        filtered_docs.sort(key=lambda x: (x.get('wirtschaft_relevance', 0), x['score']), reverse=True)
        
        return filtered_docs
    
    def _generate_wirtschaft_response(self, query: str, documents: List[Dict]) -> str:
        """Generate wirtschaft-focused response"""
        if not documents:
            return "Ich konnte keine relevanten wirtschaftlichen Informationen zu Ihrer Frage finden."
        
        key_points = []
        for doc in documents[:3]:
            text = doc['text']
            sentences = text.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in self.capability.keywords[:10]):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                key_points.extend(relevant_sentences[:2])
        
        if key_points:
            response = f"💰 **Wirtschafts-Analyse:**\n\n"
            for i, point in enumerate(key_points[:4], 1):
                if point:
                    response += f"{i}. {point}\n\n"
            
            response += f"📈 **Quelle:** Basiert auf {len(documents)} wirtschaftlichen Dokumenten"
            return response
        else:
            return f"💼 **Wirtschafts-Information:** {documents[0]['text']}"
    
    def _calculate_response_confidence(self, documents: List[Dict]) -> float:
        """Calculate confidence based on document quality and relevance"""
        if not documents:
            return 0.0
        
        avg_score = sum(doc['score'] for doc in documents) / len(documents)
        avg_relevance = sum(doc.get('wirtschaft_relevance', 0) for doc in documents) / len(documents)
        
        confidence = (avg_score * 0.7) + (avg_relevance * 0.3)
        
        return min(confidence, 0.95)

class KlimaAgent(BaseAgent):
    """Specialized agent for Klima domain queries"""
    
    def __init__(self, rag_system: AdvancedRAGSystem):
        capability = AgentCapability(
            domain="Klima",
            expertise_areas=[
                "Klimaschutz", "Energiewende", "CO2-Ziele", "Emissionen",
                "Verkehrswende", "Elektromobilität", "Erneuerbare Energien",
                "Kohleausstieg", "Atomausstieg", "Solar", "Wind", "Wasserstoff",
                "Gebäudesanierung", "Wärmepumpen", "Nachhaltigkeit", "Green Deal"
            ],
            keywords=[
                "klima", "klimaschutz", "klimawandel", "co2", "emission", "treibhausgas",
                "energie", "energiewende", "erneuerbar", "solar", "wind", "wasserstoff",
                "kohle", "atom", "gas", "öl", "verkehr", "elektro", "e-auto",
                "nachhaltigkeit", "umwelt", "umweltschutz", "green", "grün",
                "wärmepumpe", "sanierung", "effizienz", "recycling", "kreislauf"
            ],
            confidence_threshold=0.75
        )
        super().__init__(AgentType.KLIMA, capability, rag_system)
        
    async def process_query(self, query: str, context: Optional[Dict] = None) -> AgentResponse:
        """Process klima-specific queries"""
        start_time = time.time()
        
        try:
            # Enhanced query for klima domain
            enhanced_query = self._enhance_klima_query(query)
            
            # Use RAG system to get relevant documents
            documents = self.rag_system.retrieve_relevant_documents(enhanced_query, top_k=8)
            
            # Filter for klima-relevant content
            klima_docs = self._filter_klima_documents(documents)
            
            # Generate klima-focused response
            response_content = self._generate_klima_response(query, klima_docs)
            
            processing_time = time.time() - start_time
            confidence = self._calculate_response_confidence(klima_docs)
            
            response = AgentResponse(
                agent_type=self.agent_type,
                content=response_content,
                confidence=confidence,
                sources=klima_docs[:5],
                processing_time=processing_time,
                metadata={
                    'enhanced_query': enhanced_query,
                    'documents_found': len(documents),
                    'klima_documents': len(klima_docs),
                    'domain_expertise': self.capability.domain
                }
            )
            
            self.update_stats(processing_time, confidence, True)
            return response
            
        except Exception as e:
            logger.error(f"Klima Agent error: {str(e)}")
            processing_time = time.time() - start_time
            
            response = AgentResponse(
                agent_type=self.agent_type,
                content=f"Entschuldigung, ich konnte Ihre Klima-Frage nicht vollständig beantworten: {str(e)}",
                confidence=0.0,
                sources=[],
                processing_time=processing_time,
                metadata={'error': str(e)}
            )
            
            self.update_stats(processing_time, 0.0, False)
            return response
    
    def _enhance_klima_query(self, query: str) -> str:
        """Enhance query with klima-specific context"""
        klima_context = [
            "Klimaschutz", "Energiewende", "CO2-Ziele",
            "deutsche Klimapolitik", "Umweltschutz"
        ]
        
        enhanced = f"{query} {' '.join(klima_context[:2])}"
        return enhanced
    
    def _filter_klima_documents(self, documents: List[Dict]) -> List[Dict]:
        """Filter documents for klima relevance"""
        klima_keywords = self.capability.keywords
        
        filtered_docs = []
        for doc in documents:
            text_lower = doc['text'].lower()
            
            matches = sum(1 for keyword in klima_keywords if keyword in text_lower)
            
            if matches >= 2:
                doc['klima_relevance'] = matches / len(klima_keywords)
                filtered_docs.append(doc)
        
        filtered_docs.sort(key=lambda x: (x.get('klima_relevance', 0), x['score']), reverse=True)
        
        return filtered_docs
    
    def _generate_klima_response(self, query: str, documents: List[Dict]) -> str:
        """Generate klima-focused response"""
        if not documents:
            return "Ich konnte keine relevanten Klima-Informationen zu Ihrer Frage finden."
        
        key_points = []
        for doc in documents[:3]:
            text = doc['text']
            sentences = text.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in self.capability.keywords[:10]):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                key_points.extend(relevant_sentences[:2])
        
        if key_points:
            response = f"🌍 **Klima-Analyse:**\n\n"
            for i, point in enumerate(key_points[:4], 1):
                if point:
                    response += f"{i}. {point}\n\n"
            
            response += f"♻️ **Quelle:** Basiert auf {len(documents)} Klima-Dokumenten"
            return response
        else:
            return f"🌱 **Klima-Information:** {documents[0]['text']}"
    
    def _calculate_response_confidence(self, documents: List[Dict]) -> float:
        """Calculate confidence based on document quality and relevance"""
        if not documents:
            return 0.0
        
        avg_score = sum(doc['score'] for doc in documents) / len(documents)
        avg_relevance = sum(doc.get('klima_relevance', 0) for doc in documents) / len(documents)
        
        confidence = (avg_score * 0.7) + (avg_relevance * 0.3)
        
        return min(confidence, 0.95)

class QueryClassifier:
    """Intelligent query classification system for routing to appropriate agents"""
    
    def __init__(self):
        # Domain classification patterns
        self.domain_patterns = {
            'politik': {
                'keywords': [
                    'bundestag', 'bundesrat', 'gesetz', 'wahl', 'koalition', 'partei',
                    'kanzler', 'minister', 'regierung', 'opposition', 'demokratie',
                    'parlament', 'verfassung', 'grundgesetz', 'spd', 'cdu', 'grüne'
                ],
                'patterns': [
                    r'wer\s+ist\s+(kanzler|minister)',
                    r'(bundestag|bundesrat)\s+(beschluss|entscheidung)',
                    r'(wahl|abstimmung|koalition)',
                    r'(gesetz|gesetzentwurf|reform)',
                    r'(partei|opposition|regierung)'
                ]
            },
            'wirtschaft': {
                'keywords': [
                    'wirtschaft', 'geld', 'euro', 'inflation', 'arbeitsmarkt',
                    'steuer', 'haushalt', 'investition', 'bip', 'wachstum',
                    'unternehmen', 'handel', 'export', 'zinsen', 'börse'
                ],
                'patterns': [
                    r'(inflation|deflation|zinsen)',
                    r'(arbeitsmarkt|arbeitslosigkeit|job)',
                    r'(steuer|haushalt|finanzen)',
                    r'(wirtschaft|ökonomie|bip)',
                    r'(unternehmen|mittelstand|handel)'
                ]
            },
            'klima': {
                'keywords': [
                    'klima', 'co2', 'emission', 'energie', 'energiewende',
                    'erneuerbar', 'solar', 'wind', 'kohle', 'atom',
                    'verkehr', 'elektro', 'nachhaltigkeit', 'umwelt', 'grün'
                ],
                'patterns': [
                    r'(klima|klimaschutz|klimawandel)',
                    r'(energie|energiewende|erneuerbar)',
                    r'(co2|emission|treibhausgas)',
                    r'(solar|wind|wasserstoff)',
                    r'(verkehr|elektro|e-auto)'
                ]
            }
        }
    
    def classify_query(self, query: str) -> QueryClassification:
        """Classify query and determine routing strategy"""
        query_lower = query.lower()
        
        # Calculate domain scores
        domain_scores = {}
        keywords_found = []
        
        for domain, config in self.domain_patterns.items():
            # Keyword matching
            keyword_matches = [kw for kw in config['keywords'] if kw in query_lower]
            keyword_score = len(keyword_matches) / max(len(config['keywords']), 1)
            
            # Pattern matching
            pattern_matches = 0
            for pattern in config['patterns']:
                if re.search(pattern, query_lower):
                    pattern_matches += 1
            pattern_score = pattern_matches / max(len(config['patterns']), 1)
            
            # Combined score
            domain_scores[domain] = (keyword_score * 0.6) + (pattern_score * 0.4)
            
            if keyword_matches:
                keywords_found.extend(keyword_matches)
        
        # Determine primary and secondary domains
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_domain = sorted_domains[0][0] if sorted_domains[0][1] > 0.05 else 'general'
        primary_score = sorted_domains[0][1]
        
        secondary_domains = [domain for domain, score in sorted_domains[1:] if score > 0.08]
        
        # Determine complexity
        if len(secondary_domains) >= 2:
            complexity = QueryComplexity.COMPLEX
        elif len(secondary_domains) == 1 or primary_score > 0.4:
            complexity = QueryComplexity.MODERATE
        else:
            complexity = QueryComplexity.SIMPLE
        
        # Determine routing decision
        routing_decision = [AgentType.COORDINATOR]  # Always include coordinator
        
        if primary_domain == 'politik':
            routing_decision.append(AgentType.POLITIK)
        elif primary_domain == 'wirtschaft':
            routing_decision.append(AgentType.WIRTSCHAFT)
        elif primary_domain == 'klima':
            routing_decision.append(AgentType.KLIMA)
        
        # Add secondary agents for complex queries
        if complexity == QueryComplexity.COMPLEX:
            for domain in secondary_domains:
                if domain == 'politik' and AgentType.POLITIK not in routing_decision:
                    routing_decision.append(AgentType.POLITIK)
                elif domain == 'wirtschaft' and AgentType.WIRTSCHAFT not in routing_decision:
                    routing_decision.append(AgentType.WIRTSCHAFT)
                elif domain == 'klima' and AgentType.KLIMA not in routing_decision:
                    routing_decision.append(AgentType.KLIMA)
        
        # Add secondary agents for moderate complexity multi-domain queries
        elif complexity == QueryComplexity.MODERATE and len(secondary_domains) > 0:
            # Add the highest scoring secondary domain
            top_secondary = secondary_domains[0]
            if top_secondary == 'politik' and AgentType.POLITIK not in routing_decision:
                routing_decision.append(AgentType.POLITIK)
            elif top_secondary == 'wirtschaft' and AgentType.WIRTSCHAFT not in routing_decision:
                routing_decision.append(AgentType.WIRTSCHAFT)
            elif top_secondary == 'klima' and AgentType.KLIMA not in routing_decision:
                routing_decision.append(AgentType.KLIMA)
        
        return QueryClassification(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            complexity=complexity,
            keywords_found=keywords_found,
            confidence=primary_score,
            routing_decision=routing_decision
        )

class CoordinatorAgent:
    """Master coordinator agent for query routing and response synthesis"""
    
    def __init__(self, rag_system: AdvancedRAGSystem):
        self.rag_system = rag_system
        self.classifier = QueryClassifier()
        
        # Initialize specialized agents
        self.agents = {
            AgentType.POLITIK: PolitikAgent(rag_system),
            AgentType.WIRTSCHAFT: WirtschaftAgent(rag_system),
            AgentType.KLIMA: KlimaAgent(rag_system)
        }
        
        self.stats = {
            'queries_processed': 0,
            'avg_response_time': 0.0,
            'successful_responses': 0,
            'agent_usage': {agent_type.value: 0 for agent_type in AgentType},
            'complexity_distribution': {complexity.value: 0 for complexity in QueryComplexity}
        }
        
    async def process_query(self, query: str, context: Optional[Dict] = None) -> MultiAgentResponse:
        """Main entry point for multi-agent query processing"""
        start_time = time.time()
        
        try:
            # Step 1: Classify query
            classification = self.classifier.classify_query(query)
            
            # Update stats
            self.stats['complexity_distribution'][classification.complexity.value] += 1
            
            # Step 2: Route to appropriate agents
            agent_responses = []
            
            for agent_type in classification.routing_decision:
                if agent_type == AgentType.COORDINATOR:
                    continue  # Skip coordinator in agent responses
                    
                agent = self.agents.get(agent_type)
                if agent:
                    try:
                        response = await agent.process_query(query, context)
                        agent_responses.append(response)
                        self.stats['agent_usage'][agent_type.value] += 1
                    except Exception as e:
                        logger.error(f"Agent {agent_type.value} failed: {str(e)}")
            
            # Step 3: Synthesize responses
            synthesized_response = self._synthesize_responses(
                query, classification, agent_responses
            )
            
            # Update coordinator stats
            processing_time = time.time() - start_time
            self.stats['queries_processed'] += 1
            self.stats['successful_responses'] += 1
            
            current_avg = self.stats['avg_response_time']
            n = self.stats['queries_processed']
            self.stats['avg_response_time'] = ((current_avg * (n-1)) + processing_time) / n
            
            return synthesized_response
            
        except Exception as e:
            logger.error(f"Coordinator error: {str(e)}")
            
            # Fallback response
            processing_time = time.time() - start_time
            self.stats['queries_processed'] += 1
            
            return MultiAgentResponse(
                primary_response=f"Entschuldigung, ich konnte Ihre Frage nicht vollständig bearbeiten: {str(e)}",
                contributing_agents=[],
                confidence=0.0,
                sources=[],
                synthesis_metadata={'error': str(e)},
                total_processing_time=processing_time
            )
    
    def _synthesize_responses(self, query: str, classification: QueryClassification, 
                            responses: List[AgentResponse]) -> MultiAgentResponse:
        """Synthesize multiple agent responses into coherent final response"""
        
        if not responses:
            # Fallback to general RAG if no agent responses
            try:
                documents = self.rag_system.retrieve_relevant_documents(query, top_k=5)
                fallback_content = documents[0]['text'] if documents else "Keine Informationen gefunden."
                
                return MultiAgentResponse(
                    primary_response=f"📋 **Allgemeine Information:** {fallback_content}",
                    contributing_agents=[],
                    confidence=0.5,
                    sources=documents[:3],
                    synthesis_metadata={'fallback': True, 'classification': asdict(classification)},
                    total_processing_time=0.0
                )
            except Exception:
                return MultiAgentResponse(
                    primary_response="Entschuldigung, ich konnte keine relevanten Informationen finden.",
                    contributing_agents=[],
                    confidence=0.0,
                    sources=[],
                    synthesis_metadata={'fallback_failed': True},
                    total_processing_time=0.0
                )
        
        # Single agent response
        if len(responses) == 1:
            response = responses[0]
            return MultiAgentResponse(
                primary_response=response.content,
                contributing_agents=[response.agent_type],
                confidence=response.confidence,
                sources=response.sources,
                synthesis_metadata={
                    'single_agent': True,
                    'agent_metadata': response.metadata,
                    'classification': asdict(classification)
                },
                total_processing_time=response.processing_time
            )
        
        # Multi-agent synthesis
        responses.sort(key=lambda x: x.confidence, reverse=True)  # Sort by confidence
        
        # Build synthesized response
        synthesized_content = "🤖 **Multi-Agent Analyse:**\n\n"
        
        contributing_agents = []
        all_sources = []
        total_confidence = 0.0
        
        for i, response in enumerate(responses):
            agent_emoji = {
                AgentType.POLITIK: "🏛️",
                AgentType.WIRTSCHAFT: "💰", 
                AgentType.KLIMA: "🌍"
            }
            
            emoji = agent_emoji.get(response.agent_type, "🤖")
            domain = response.agent_type.value.title()
            
            synthesized_content += f"{emoji} **{domain}-Perspektive:**\n"
            synthesized_content += f"{response.content}\n\n"
            
            contributing_agents.append(response.agent_type)
            all_sources.extend(response.sources)
            total_confidence += response.confidence
        
        # Add synthesis summary
        avg_confidence = total_confidence / len(responses)
        synthesized_content += f"📊 **Zusammenfassung:** Analyse von {len(responses)} Fachbereichen "
        synthesized_content += f"mit durchschnittlicher Konfidenz von {avg_confidence:.1%}"
        
        # Remove duplicate sources
        unique_sources = []
        seen_sources = set()
        for source in all_sources:
            source_id = source.get('text', '')[:50]  # Use first 50 chars as ID
            if source_id not in seen_sources:
                unique_sources.append(source)
                seen_sources.add(source_id)
        
        total_processing_time = sum(r.processing_time for r in responses)
        
        return MultiAgentResponse(
            primary_response=synthesized_content,
            contributing_agents=contributing_agents,
            confidence=avg_confidence,
            sources=unique_sources[:8],  # Top 8 unique sources
            synthesis_metadata={
                'multi_agent': True,
                'agent_count': len(responses),
                'classification': asdict(classification),
                'individual_confidences': [r.confidence for r in responses]
            },
            total_processing_time=total_processing_time
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and statistics"""
        agent_stats = {}
        for agent_type, agent in self.agents.items():
            agent_stats[agent_type.value] = agent.stats
        
        return {
            'coordinator_stats': self.stats,
            'agent_stats': agent_stats,
            'system_info': {
                'total_agents': len(self.agents),
                'rag_system_type': type(self.rag_system).__name__,
                'classification_domains': list(self.classifier.domain_patterns.keys())
            }
        }

class MultiAgentSystem:
    """Main interface for the Multi-Agent Intelligence System"""
    
    def __init__(self, rag_system: Optional[AdvancedRAGSystem] = None):
        """Initialize multi-agent system with optional RAG system"""
        
        if rag_system is None:
            # Initialize default RAG system
            try:
                self.rag_system = AdvancedRAGSystem()
            except Exception as e:
                logger.error(f"Failed to initialize RAG system: {e}")
                raise
        else:
            self.rag_system = rag_system
        
        # Initialize coordinator
        self.coordinator = CoordinatorAgent(self.rag_system)
        
        logger.info("🚀 Multi-Agent Intelligence System initialized successfully")
    
    async def query(self, question: str, context: Optional[Dict] = None) -> MultiAgentResponse:
        """Main query interface"""
        return await self.coordinator.process_query(question, context)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return self.coordinator.get_system_status()
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get information about agent capabilities"""
        capabilities = {}
        
        for agent_type, agent in self.coordinator.agents.items():
            capabilities[agent_type.value] = {
                'domain': agent.capability.domain,
                'expertise_areas': agent.capability.expertise_areas,
                'keyword_count': len(agent.capability.keywords),
                'confidence_threshold': agent.capability.confidence_threshold
            }
        
        return capabilities

# Export main classes
__all__ = [
    'MultiAgentSystem',
    'AgentType',
    'QueryComplexity', 
    'MultiAgentResponse',
    'CoordinatorAgent',
    'PolitikAgent',
    'WirtschaftAgent',
    'KlimaAgent'
]