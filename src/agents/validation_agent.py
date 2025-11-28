"""
Validation Agent - Sequential quality assurance

SPECIALIZATION: Quality control
PATTERN: Sequential execution
CHECKS: Facts, sources, consistency
"""

from datetime import datetime
import logging

from src.agents.base_agent import BaseAgent
from src.orchestrator.task_models import ResearchResult

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """
    Validation agent for sequential quality checks
    
    ROLE IN SYSTEM:
    ---------------
    Processes research results sequentially to ensure
    quality before synthesis. Acts as quality gate.
    
    VALIDATION CRITERIA:
    --------------------
    1. Source credibility (40%)
    2. Content quality (30%)
    3. Confidence alignment (30%)
    
    THRESHOLD: 0.75
    Below threshold → Needs revision
    Above threshold → Approved
    """
    
    async def execute_task(self, result_to_validate: ResearchResult) -> ResearchResult:
        """
        Validate research result
        
        PROCESS:
        --------
        1. Check source credibility
        2. Assess content quality
        3. Verify confidence alignment
        4. Generate validation report
        5. Provide improvement recommendations
        
        WHY sequential?
        - Each result validated independently
        - Quality gate before synthesis
        - Prevents cascade of errors
        """
        start_time = datetime.now()
        
        logger.info(
            f"{self.agent.name} validating result from {result_to_validate.agent_id}"
        )
        
        try:
            # Validation checks
            source_score = self._check_sources(result_to_validate.sources)
            content_score = self._check_content(result_to_validate.content)
            confidence_score = result_to_validate.confidence
            
            # Calculate overall validation score
            validation_score = (
                source_score * 0.4 +
                content_score * 0.3 +
                confidence_score * 0.3
            )
            
            is_validated = validation_score >= 0.75
            
            # Generate detailed report
            report = self._generate_report(
                validation_score,
                source_score,
                content_score,
                confidence_score,
                is_validated
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create validation result
            result = ResearchResult(
                task_id=result_to_validate.task_id,
                agent_id=self.agent.id,
                content=report,
                sources=['Validation Framework'],
                confidence=validation_score,
                validated=is_validated,
                validation_notes=f"Score: {validation_score:.2f}",
                metadata={
                    'source_score': source_score,
                    'content_score': content_score,
                    'execution_time': execution_time,
                    'original_agent': result_to_validate.agent_id
                }
            )
            
            # Store validation experience
            await self.store_experience(
                f"Validated result with score {validation_score:.2f}. "
                f"Status: {'PASS' if is_validated else 'FAIL'}",
                importance=0.8
            )
            
            self.update_metrics(execution_time, success=True)
            
            logger.info(
                f"{self.agent.name} completed: "
                f"{'VALIDATED' if is_validated else 'REJECTED'} "
                f"(score: {validation_score:.2f})"
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"{self.agent.name} failed: {str(e)}", exc_info=True)
            
            self.update_metrics(execution_time, success=False)
            
            return ResearchResult(
                task_id=result_to_validate.task_id,
                agent_id=self.agent.id,
                content=f"Validation failed: {str(e)}",
                sources=[],
                confidence=0.0,
                validated=False,
                metadata={'error': str(e)}
            )
    
    def _check_sources(self, sources: list) -> float:
        """
        Score source credibility
        
        HEURISTIC:
        ----------
        More sources = higher credibility
        
        SCORING:
        - 0 sources: 0.3 (poor)
        - 1-2 sources: 0.5-0.6 (fair)
        - 3-4 sources: 0.7-0.8 (good)
        - 5+ sources: 0.9+ (excellent)
        
        PRODUCTION:
        Check against whitelist of trusted sources
        """
        if not sources:
            return 0.3
        
        # Base score
        score = 0.5
        
        # Add bonus for each source (diminishing returns)
        for i in range(len(sources)):
            bonus = 0.1 / (i + 1)  # 0.1, 0.05, 0.033, ...
            score += bonus
        
        return min(1.0, score)
    
    def _check_content(self, content: str) -> float:
        """
        Score content quality
        
        METRICS:
        --------
        1. Length (comprehensive?)
        2. Structure (organized?)
        3. Depth (detailed?)
        
        SCORING:
        - <100 words: 0.3
        - 100-200 words: 0.5
        - 200-300 words: 0.7
        - >300 words: 0.8
        + bonus for structure
        """
        score = 0.5
        
        word_count = len(content.split())
        
        # Length scoring
        if word_count < 100:
            length_score = 0.3
        elif word_count < 200:
            length_score = 0.5
        elif word_count < 300:
            length_score = 0.7
        else:
            length_score = 0.8
        
        score = length_score
        
        # Structure bonus
        structure_indicators = [
            '\n\n',      # Paragraphs
            '1.',        # Numbered lists
            '-',         # Bullet points
            '#',         # Headers
        ]
        
        if any(ind in content for ind in structure_indicators):
            score += 0.2
        
        return min(1.0, score)
    
    def _generate_report(
        self,
        overall: float,
        sources: float,
        content: float,
        confidence: float,
        validated: bool
    ) -> str:
        """Generate comprehensive validation report"""
        
        status = "✓ VALIDATED" if validated else "✗ NEEDS REVISION"
        
        report = f"""╔═══════════════════════════════════════════════════════════╗
║               VALIDATION REPORT                           ║
╚═══════════════════════════════════════════════════════════╝

OVERALL SCORE: {overall:.2f} / 1.00
STATUS: {status}

DETAILED SCORES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Source Credibility:  {sources:.2f} / 1.00  (40% weight)
- Content Quality:     {content:.2f} / 1.00  (30% weight)
- Confidence Level:    {confidence:.2f} / 1.00  (30% weight)

RECOMMENDATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{self._generate_recommendations(overall, sources, content, confidence)}

QUALITY GATE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Threshold: 0.75
Result: {"PASS - Ready for synthesis" if validated else "FAIL - Revision required"}
"""
        return report
    
    def _generate_recommendations(
        self,
        overall: float,
        sources: float,
        content: float,
        confidence: float
    ) -> str:
        """Generate actionable recommendations"""
        recs = []
        
        if sources < 0.7:
            recs.append("→ Add more diverse and credible sources")
        
        if content < 0.7:
            recs.append("→ Improve content structure and depth")
            recs.append("→ Add more detailed explanations")
        
        if confidence < 0.7:
            recs.append("→ Verify claims with additional sources")
            recs.append("→ Strengthen factual grounding")
        
        if overall >= 0.9:
            recs.append("→ Excellent quality! Ready for immediate use")
        elif overall >= 0.75:
            recs.append("→ Good quality. Minor improvements possible")
        else:
            recs.append("→ Significant revision required")
            recs.append("→ Focus on areas scoring below 0.7")
        
        return '\n'.join(recs) if recs else "→ No specific recommendations"
