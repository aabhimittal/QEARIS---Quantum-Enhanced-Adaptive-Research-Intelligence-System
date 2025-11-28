"""
Agent Evaluator - Performance assessment and recommendations

PURPOSE: Evaluate agent performance and provide improvement recommendations
METRICS: Success rate, speed, quality, consistency
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import statistics
import logging

from src.orchestrator.task_models import Agent, ResearchResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Agent evaluation result"""
    agent_id: str
    overall_score: float
    metrics: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    grade: str  # A, B, C, D, F


class AgentEvaluator:
    """
    Evaluate agent performance
    
    EVALUATION DIMENSIONS:
    ----------------------
    1. Success Rate (30%)
    2. Speed/Efficiency (25%)
    3. Quality (25%)
    4. Consistency (20%)
    
    SCORING:
    --------
    A: 90-100 (Excellent)
    B: 80-89  (Good)
    C: 70-79  (Fair)
    D: 60-69  (Poor)
    F: <60    (Failing)
    """
    
    def __init__(self):
        self.agent_history: Dict[str, List[ResearchResult]] = {}
        logger.info("AgentEvaluator initialized")
    
    def record_result(self, agent_id: str, result: ResearchResult):
        """Record agent result for evaluation"""
        if agent_id not in self.agent_history:
            self.agent_history[agent_id] = []
        
        self.agent_history[agent_id].append(result)
    
    def evaluate_agent(
        self,
        agent: Agent,
        results: List[ResearchResult] = None
    ) -> EvaluationResult:
        """
        Evaluate single agent
        
        PROCESS:
        --------
        1. Collect metrics
        2. Calculate scores
        3. Identify strengths/weaknesses
        4. Generate recommendations
        5. Assign grade
        """
        # Use provided results or history
        if results is None:
            results = self.agent_history.get(agent.id, [])
        
        if not results:
            return self._create_no_data_evaluation(agent.id)
        
        # Calculate metrics
        success_score = self._calculate_success_rate(results)
        speed_score = self._calculate_speed_score(results)
        quality_score = self._calculate_quality_score(results)
        consistency_score = self._calculate_consistency_score(results)
        
        # Weighted overall score
        overall_score = (
            success_score * 0.30 +
            speed_score * 0.25 +
            quality_score * 0.25 +
            consistency_score * 0.20
        )
        
        # Identify strengths and weaknesses
        scores = {
            'success_rate': success_score,
            'speed': speed_score,
            'quality': quality_score,
            'consistency': consistency_score
        }
        
        strengths = self._identify_strengths(scores)
        weaknesses = self._identify_weaknesses(scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            scores,
            weaknesses,
            agent.specialization
        )
        
        # Assign grade
        grade = self._assign_grade(overall_score)
        
        evaluation = EvaluationResult(
            agent_id=agent.id,
            overall_score=overall_score,
            metrics=scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            grade=grade
        )
        
        logger.info(
            f"Evaluated {agent.id}: "
            f"Score={overall_score:.2f}, Grade={grade}"
        )
        
        return evaluation
    
    def _calculate_success_rate(self, results: List[ResearchResult]) -> float:
        """
        Calculate success rate
        
        FORMULA:
        --------
        success_rate = successful_results / total_results
        
        SUCCESS CRITERIA:
        - Confidence > 0.7
        - Has sources
        - No errors
        """
        if not results:
            return 0.0
        
        successful = sum(
            1 for r in results
            if r.confidence > 0.7 and r.sources and 'error' not in r.metadata
        )
        
        return successful / len(results)
    
    def _calculate_speed_score(self, results: List[ResearchResult]) -> float:
        """
        Calculate speed score
        
        FORMULA:
        --------
        Normalize execution time against target
        Target: 30s
        Faster = better (up to 1.0)
        """
        times = [
            r.metadata.get('execution_time', 30)
            for r in results
        ]
        
        if not times:
            return 0.5
        
        avg_time = statistics.mean(times)
        target_time = 30.0  # seconds
        
        # Score: 1.0 if at or below target, decreasing linearly
        if avg_time <= target_time:
            return 1.0
        else:
            return max(0.0, 1.0 - (avg_time - target_time) / target_time)
    
    def _calculate_quality_score(self, results: List[ResearchResult]) -> float:
        """
        Calculate quality score
        
        METRICS:
        --------
        - Average confidence
        - Source count
        - Content length
        """
        if not results:
            return 0.0
        
        # Confidence
        avg_confidence = statistics.mean([r.confidence for r in results])
        
        # Sources
        avg_sources = statistics.mean([len(r.sources) for r in results])
        source_score = min(avg_sources / 5.0, 1.0)
        
        # Content length
        avg_length = statistics.mean([len(r.content) for r in results])
        length_score = min(avg_length / 1000.0, 1.0)
        
        # Combined quality
        quality = (
            avg_confidence * 0.5 +
            source_score * 0.3 +
            length_score * 0.2
        )
        
        return quality
    
    def _calculate_consistency_score(self, results: List[ResearchResult]) -> float:
        """
        Calculate consistency score
        
        METRIC:
        -------
        Low variance in confidence = high consistency
        """
        if len(results) < 2:
            return 1.0
        
        confidences = [r.confidence for r in results]
        variance = statistics.variance(confidences)
        
        # Low variance = high consistency
        consistency = max(0.0, 1.0 - variance)
        
        return consistency
    
    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """Identify strengths (scores > 0.8)"""
        strengths = []
        
        for metric, score in scores.items():
            if score > 0.8:
                strengths.append(f"Strong {metric.replace('_', ' ')}")
        
        return strengths
    
    def _identify_weaknesses(self, scores: Dict[str, float]) -> List[str]:
        """Identify weaknesses (scores < 0.6)"""
        weaknesses = []
        
        for metric, score in scores.items():
            if score < 0.6:
                weaknesses.append(f"Weak {metric.replace('_', ' ')}")
        
        return weaknesses
    
    def _generate_recommendations(
        self,
        scores: Dict[str, float],
        weaknesses: List[str],
        specialization: str
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Success rate recommendations
        if scores['success_rate'] < 0.7:
            recommendations.append(
                "Improve error handling and retry logic"
            )
            recommendations.append(
                "Verify source credibility before using"
            )
        
        # Speed recommendations
        if scores['speed'] < 0.7:
            recommendations.append(
                "Optimize tool usage and reduce redundant searches"
            )
            recommendations.append(
                "Cache frequently accessed information"
            )
        
        # Quality recommendations
        if scores['quality'] < 0.7:
            recommendations.append(
                "Use more diverse and credible sources"
            )
            recommendations.append(
                f"Deepen expertise in {specialization} domain"
            )
        
        # Consistency recommendations
        if scores['consistency'] < 0.7:
            recommendations.append(
                "Standardize research methodology"
            )
            recommendations.append(
                "Improve prompt engineering for stable outputs"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "Maintain current performance level"
            )
            recommendations.append(
                "Consider taking on more complex tasks"
            )
        
        return recommendations
    
    def _assign_grade(self, score: float) -> str:
        """Assign letter grade"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _create_no_data_evaluation(self, agent_id: str) -> EvaluationResult:
        """Create evaluation when no data available"""
        return EvaluationResult(
            agent_id=agent_id,
            overall_score=0.0,
            metrics={},
            strengths=[],
            weaknesses=["No performance data available"],
            recommendations=["Complete tasks to generate evaluation data"],
            grade="N/A"
        )
    
    def generate_report(self, evaluation: EvaluationResult) -> str:
        """Generate human-readable evaluation report"""
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           AGENT PERFORMANCE EVALUATION                       ║
╚══════════════════════════════════════════════════════════════╝

Agent ID: {evaluation.agent_id}
Overall Score: {evaluation.overall_score:.2f} / 1.00
Grade: {evaluation.grade}

PERFORMANCE METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for metric, score in evaluation.metrics.items():
            bar = "█" * int(score * 20)
            report += f"{metric.replace('_', ' ').title():20s} [{score:.2f}] {bar}\n"
        
        report += "\nSTRENGTHS:\n"
        report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for strength in evaluation.strengths:
            report += f"✓ {strength}\n"
        
        report += "\nWEAKNESSES:\n"
        report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for weakness in evaluation.weaknesses:
            report += f"✗ {weakness}\n"
        
        report += "\nRECOMMENDATIONS:\n"
        report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for i, rec in enumerate(evaluation.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report
