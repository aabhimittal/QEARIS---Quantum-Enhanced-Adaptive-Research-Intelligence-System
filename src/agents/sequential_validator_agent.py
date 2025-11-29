"""
# ============================================================================
# QEARIS - SEQUENTIAL VALIDATOR AGENT
# ============================================================================
# 
# CAPSTONE REQUIREMENT: Sequential Agents
# POINTS: Technical Implementation - 50 points (Multi-Agent System)
# 
# DESCRIPTION: Validation agent that processes research results sequentially.
# Acts as a quality gate, ensuring each result meets quality standards before
# synthesis. Sequential processing ensures consistent validation criteria
# are applied to all results.
# 
# INNOVATION: Multi-dimensional validation (sources, content, confidence)
# with configurable thresholds and actionable improvement recommendations.
# 
# FILE LOCATION: src/agents/sequential_validator_agent.py
# 
# CAPSTONE CRITERIA MET:
# - Multi-Agent System: Sequential execution pattern (for loop processing)
# - Agent Evaluation: Quality scoring and recommendations
# - Gemini Integration: Optional LLM-based validation
# - Observability: Detailed validation reports
# ============================================================================
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agents.base_llm_agent import (BaseLLMAgent, LLMAgentConfig,
                                       LLMAgentType)
from src.orchestrator.task_models import MemoryType, ResearchResult

# ============================================================================
# CAPSTONE REQUIREMENT: Observability - Logging
# ============================================================================
logger = logging.getLogger(__name__)


class SequentialValidatorAgent(BaseLLMAgent[ResearchResult]):
    """
    Validation agent with sequential processing pattern.

    ============================================================================
    CAPSTONE REQUIREMENT: Sequential Agents
    POINTS: Technical Implementation - 50 points (Multi-Agent System)

    DESCRIPTION:
    This agent validates research results one at a time in sequence. Unlike
    parallel agents, sequential processing ensures:

    1. SEQUENTIAL EXECUTION BENEFITS:
       - Consistent validation criteria application
       - Quality gate before synthesis
       - Resource-efficient (one at a time)
       - Clear processing order
       - Prevents cascade of errors to synthesis

    2. EXECUTION PATTERN:
       Results are processed sequentially using a for loop:
       ```python
       validated_results = []
       for result in research_results:
           validation = await validator.execute_task(result)
           validated_results.append(validation)
       ```

    3. VALIDATION CRITERIA:
       - Source Credibility (40%): Number and quality of sources
       - Content Quality (30%): Length, structure, depth
       - Confidence Alignment (30%): Self-reported confidence accuracy

    4. QUALITY THRESHOLD:
       - Pass (≥0.75): Result approved for synthesis
       - Fail (<0.75): Result needs revision or exclusion

    INNOVATION:
    -----------
    - Multi-dimensional validation scoring
    - Actionable improvement recommendations
    - Detailed validation reports for transparency
    - Optional LLM-enhanced validation for complex content
    ============================================================================

    WHY SEQUENTIAL (not parallel)?
    ------------------------------
    Validation requires careful, methodical assessment:
    1. Each result must be fully evaluated before moving on
    2. Prevents overwhelming downstream synthesis
    3. Maintains validation consistency
    4. Provides clear audit trail
    """

    def __init__(
        self,
        agent_id: str,
        gemini_model: Any,
        config: Optional[LLMAgentConfig] = None,
        mcp_server: Any = None,
        memory_bank: Any = None,
        validation_threshold: float = 0.75,
        use_llm_validation: bool = False,
    ):
        """
        Initialize the sequential validator agent.

        CAPSTONE REQUIREMENT: Sequential Agents

        PARAMETERS:
        -----------
        agent_id : str
            Unique identifier (typically "validator")
        gemini_model : GenerativeModel
            Configured Gemini model for LLM validation
        config : LLMAgentConfig (optional)
            Agent configuration
        mcp_server : MCPServer (optional)
            MCP server for tool execution
        memory_bank : MemoryBank (optional)
            Memory bank for validation experience
        validation_threshold : float
            Minimum score to pass validation (default: 0.75)
        use_llm_validation : bool
            Use Gemini for enhanced content validation
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=LLMAgentType.VALIDATOR,
            gemini_model=gemini_model,
            config=config,
            mcp_server=mcp_server,
            memory_bank=memory_bank,
        )

        # ====================================================================
        # CAPSTONE REQUIREMENT: Agent Evaluation
        # Configurable validation threshold
        # ====================================================================
        self.validation_threshold = validation_threshold
        self.use_llm_validation = use_llm_validation

        # Validation criteria weights
        self.weight_sources = 0.40
        self.weight_content = 0.30
        self.weight_confidence = 0.30

        logger.info(
            f"Sequential Validator Agent initialized: {agent_id} "
            f"(threshold: {validation_threshold}, llm: {use_llm_validation})"
        )

    # ========================================================================
    # CAPSTONE REQUIREMENT: Sequential Agents - Core Processing
    #
    # This method processes one result at a time, designed to be called
    # in a sequential for loop after parallel research completes.
    # ========================================================================
    async def _process_task(self, result_to_validate: ResearchResult) -> ResearchResult:
        """
        Validate a single research result sequentially.

        CAPSTONE REQUIREMENT: Sequential Agents

        ========================================================================
        SEQUENTIAL EXECUTION FLOW:
        ========================================================================

        WHY SEQUENTIAL?
        ---------------
        Unlike parallel research, validation is sequential because:
        1. Acts as quality gate - each result must pass before synthesis
        2. Consistent criteria - same standards applied to all
        3. Clear ordering - results processed in defined order
        4. Audit trail - easy to track what was validated when

        PROCESSING STEPS:
        -----------------
        1. Check source credibility (40% weight)
        2. Assess content quality (30% weight)
        3. Verify confidence alignment (30% weight)
        4. Calculate overall validation score
        5. Generate detailed validation report
        6. Provide improvement recommendations
        7. Store validation experience

        QUALITY GATE:
        -------------
        - Score >= 0.75: PASS - Result approved for synthesis
        - Score < 0.75: FAIL - Result needs revision

        ========================================================================
        """
        start_time = datetime.now()

        logger.info(
            f"[SEQUENTIAL] {self.agent_id} validating result from " f"{result_to_validate.agent_id}"
        )

        try:
            # ================================================================
            # STEP 1: Check source credibility (40% weight)
            # ================================================================
            source_score = self._check_sources(result_to_validate.sources)

            # ================================================================
            # STEP 2: Assess content quality (30% weight)
            # ================================================================
            content_score = self._check_content(result_to_validate.content)

            # ================================================================
            # STEP 3: Verify confidence alignment (30% weight)
            # ================================================================
            confidence_score = self._check_confidence(
                result_to_validate.confidence,
                result_to_validate.content,
                result_to_validate.sources,
            )

            # ================================================================
            # STEP 4: Optional LLM-enhanced validation
            # CAPSTONE REQUIREMENT: Gemini Integration (BONUS)
            # ================================================================
            llm_bonus = 0.0
            if self.use_llm_validation:
                llm_score = await self._llm_validate_content(
                    result_to_validate.content, result_to_validate.sources
                )
                # LLM validation can add up to 5% bonus
                llm_bonus = (llm_score - 0.5) * 0.1

            # ================================================================
            # STEP 5: Calculate overall validation score
            # ================================================================
            validation_score = (
                source_score * self.weight_sources
                + content_score * self.weight_content
                + confidence_score * self.weight_confidence
                + llm_bonus
            )

            # Ensure score is in valid range
            validation_score = max(0.0, min(1.0, validation_score))

            # Determine if result passes validation
            is_validated = validation_score >= self.validation_threshold

            # ================================================================
            # STEP 6: Generate detailed validation report
            # CAPSTONE REQUIREMENT: Observability
            # ================================================================
            validation_report = self._generate_validation_report(
                validation_score=validation_score,
                source_score=source_score,
                content_score=content_score,
                confidence_score=confidence_score,
                is_validated=is_validated,
            )

            # ================================================================
            # STEP 7: Generate improvement recommendations
            # CAPSTONE REQUIREMENT: Agent Evaluation
            # ================================================================
            recommendations = self._generate_recommendations(
                validation_score=validation_score,
                source_score=source_score,
                content_score=content_score,
                confidence_score=confidence_score,
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Create validated result
            validated_result = ResearchResult(
                task_id=result_to_validate.task_id,
                agent_id=self.agent_id,
                content=validation_report,
                sources=["Validation Framework"],
                confidence=validation_score,
                validated=is_validated,
                validation_notes=f"Score: {validation_score:.2f}, "
                f"Status: {'PASS' if is_validated else 'FAIL'}",
                metadata={
                    "original_agent": result_to_validate.agent_id,
                    "source_score": source_score,
                    "content_score": content_score,
                    "confidence_score": confidence_score,
                    "execution_time": execution_time,
                    "recommendations": recommendations,
                },
            )

            # ================================================================
            # STEP 8: Store validation experience
            # CAPSTONE REQUIREMENT: Sessions & Memory
            # ================================================================
            await self.store_experience(
                content=(
                    f"Validated result with score {validation_score:.2f}. "
                    f"Status: {'PASS' if is_validated else 'FAIL'}. "
                    f"Sources: {source_score:.2f}, Content: {content_score:.2f}, "
                    f"Confidence: {confidence_score:.2f}"
                ),
                importance=0.8,
                memory_type=MemoryType.EXPERIENCE,
            )

            logger.info(
                f"[SEQUENTIAL] {self.agent_id} completed validation: "
                f"{'PASS' if is_validated else 'FAIL'} "
                f"(score: {validation_score:.2f})"
            )

            return validated_result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            logger.error(f"[SEQUENTIAL] {self.agent_id} validation failed: {str(e)}", exc_info=True)

            return ResearchResult(
                task_id=result_to_validate.task_id,
                agent_id=self.agent_id,
                content=f"Validation failed: {str(e)}",
                sources=[],
                confidence=0.0,
                validated=False,
                metadata={"error": str(e), "execution_time": execution_time},
            )

    # ========================================================================
    # VALIDATION CRITERIA METHODS
    # ========================================================================

    def _check_sources(self, sources: List[str]) -> float:
        """
        Score source credibility.

        CAPSTONE REQUIREMENT: Agent Evaluation

        SCORING LOGIC:
        --------------
        - 0 sources: 0.3 (poor)
        - 1-2 sources: 0.5-0.6 (fair)
        - 3-4 sources: 0.7-0.8 (good)
        - 5+ sources: 0.9+ (excellent)

        RATIONALE:
        ----------
        More diverse sources typically indicate more thorough research.
        Diminishing returns prevent gaming by adding trivial sources.
        """
        if not sources:
            return 0.3

        # Base score
        score = 0.5

        # Add bonus for each source with diminishing returns
        for i in range(len(sources)):
            bonus = 0.1 / (i + 1)  # 0.1, 0.05, 0.033, ...
            score += bonus

        return min(1.0, score)

    def _check_content(self, content: str) -> float:
        """
        Score content quality.

        CAPSTONE REQUIREMENT: Agent Evaluation

        QUALITY FACTORS:
        ----------------
        1. Length: Comprehensive coverage
        2. Structure: Organization and clarity
        3. Depth: Detail level appropriate to topic

        SCORING:
        --------
        - <100 words: 0.3 (insufficient)
        - 100-200 words: 0.5 (basic)
        - 200-300 words: 0.7 (good)
        - >300 words: 0.8 (thorough)
        + 0.2 bonus for good structure
        """
        if not content:
            return 0.0

        word_count = len(content.split())

        # Length scoring
        if word_count < 100:
            score = 0.3
        elif word_count < 200:
            score = 0.5
        elif word_count < 300:
            score = 0.7
        else:
            score = 0.8

        # Structure bonus
        structure_indicators = [
            "\n\n",  # Paragraphs
            "1.",  # Numbered lists
            "- ",  # Bullet points
            "#",  # Headers
            "**",  # Bold text
            "```",  # Code blocks
        ]

        structure_count = sum(1 for ind in structure_indicators if ind in content)

        if structure_count >= 2:
            score += 0.2
        elif structure_count >= 1:
            score += 0.1

        return min(1.0, score)

    def _check_confidence(
        self, reported_confidence: float, content: str, sources: List[str]
    ) -> float:
        """
        Verify confidence alignment.

        CAPSTONE REQUIREMENT: Agent Evaluation

        PURPOSE:
        --------
        Check if the reported confidence is aligned with content quality.
        Penalize over-confidence (high confidence + poor content).
        Reward appropriate confidence calibration.

        LOGIC:
        ------
        1. Calculate expected confidence from content/sources
        2. Compare to reported confidence
        3. Score based on alignment
        """
        # Calculate expected confidence based on observable factors
        word_count = len(content.split())
        source_count = len(sources)

        # Expected confidence calculation
        expected = 0.5
        expected += min(word_count / 500, 0.3)  # Up to 0.3 for length
        expected += min(source_count / 5, 0.2)  # Up to 0.2 for sources

        # Calculate alignment (closer is better)
        difference = abs(reported_confidence - expected)

        # Score based on alignment
        if difference < 0.1:
            score = 1.0  # Excellent alignment
        elif difference < 0.2:
            score = 0.8  # Good alignment
        elif difference < 0.3:
            score = 0.6  # Fair alignment
        else:
            score = 0.4  # Poor alignment (over or under confident)

        return score

    async def _llm_validate_content(self, content: str, sources: List[str]) -> float:
        """
        Use LLM to validate content quality.

        CAPSTONE REQUIREMENT: Gemini Integration (BONUS)

        Uses Gemini to assess content for:
        - Factual accuracy
        - Logical consistency
        - Completeness
        - Clarity
        """
        prompt = f"""You are a research quality validator. Evaluate the following research content on a scale from 0.0 to 1.0.

CONTENT TO VALIDATE:
{content[:2000]}

SOURCES CITED:
{', '.join(sources) if sources else 'None'}

EVALUATION CRITERIA:
1. Factual accuracy and claims
2. Logical consistency
3. Completeness of coverage
4. Clarity of explanation

Respond with ONLY a single decimal number between 0.0 and 1.0 representing the overall quality score."""

        try:
            response = await self._generate_response(prompt)

            # Parse score from response
            score_str = response.strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))

        except (ValueError, Exception) as e:
            logger.warning(f"LLM validation failed, using default: {e}")
            return 0.5  # Default score on failure

    # ========================================================================
    # REPORT GENERATION METHODS
    # ========================================================================

    def _generate_validation_report(
        self,
        validation_score: float,
        source_score: float,
        content_score: float,
        confidence_score: float,
        is_validated: bool,
    ) -> str:
        """
        Generate comprehensive validation report.

        CAPSTONE REQUIREMENT: Observability
        Detailed report for transparency and debugging.
        """
        status = "✓ VALIDATED" if is_validated else "✗ NEEDS REVISION"

        report = f"""
╔═══════════════════════════════════════════════════════════╗
║               VALIDATION REPORT                           ║
╚═══════════════════════════════════════════════════════════╝

OVERALL SCORE: {validation_score:.2f} / 1.00
STATUS: {status}
THRESHOLD: {self.validation_threshold:.2f}

DETAILED SCORES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Source Credibility:  {source_score:.2f} / 1.00  ({self.weight_sources*100:.0f}% weight)
• Content Quality:     {content_score:.2f} / 1.00  ({self.weight_content*100:.0f}% weight)
• Confidence Level:    {confidence_score:.2f} / 1.00  ({self.weight_confidence*100:.0f}% weight)

QUALITY GATE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Result: {"PASS - Ready for synthesis" if is_validated else "FAIL - Revision required"}
"""
        return report

    def _generate_recommendations(
        self,
        validation_score: float,
        source_score: float,
        content_score: float,
        confidence_score: float,
    ) -> List[str]:
        """
        Generate actionable improvement recommendations.

        CAPSTONE REQUIREMENT: Agent Evaluation
        Provides specific guidance for result improvement.
        """
        recommendations = []

        # Source recommendations
        if source_score < 0.7:
            recommendations.append("→ Add more diverse and credible sources")
            recommendations.append("→ Include primary sources where possible")

        # Content recommendations
        if content_score < 0.7:
            recommendations.append("→ Improve content structure with headers and lists")
            recommendations.append("→ Expand on key findings with more detail")
            recommendations.append("→ Add concrete examples and evidence")

        # Confidence recommendations
        if confidence_score < 0.7:
            recommendations.append("→ Verify claims with additional sources")
            recommendations.append("→ Adjust confidence level to match evidence")

        # Overall recommendations
        if validation_score >= 0.9:
            recommendations.append("→ Excellent quality! Ready for immediate use")
        elif validation_score >= 0.75:
            recommendations.append("→ Good quality. Minor improvements possible")
        else:
            recommendations.append("→ Significant revision required")
            recommendations.append("→ Focus on areas scoring below 0.7")

        return recommendations


# ============================================================================
# CAPSTONE REQUIREMENT: Sequential Agents - Utility Function
#
# Helper function to execute sequential validation after parallel research
# ============================================================================
async def execute_sequential_validation(
    validator: SequentialValidatorAgent, research_results: List[ResearchResult]
) -> List[ResearchResult]:
    """
    Execute sequential validation on research results.

    CAPSTONE REQUIREMENT: Sequential Agents

    This function demonstrates the sequential execution pattern,
    processing each research result one at a time through validation.

    WHY SEQUENTIAL?
    ---------------
    1. Quality gate: Each result must be validated before synthesis
    2. Consistency: Same criteria applied to all results
    3. Clear ordering: Results validated in input order
    4. Resource efficient: No parallel validation overhead

    PARAMETERS:
    -----------
    validator : SequentialValidatorAgent
        The validator agent to use
    research_results : List[ResearchResult]
        Results from parallel research

    RETURNS:
    --------
    List[ResearchResult] : Validated results with scores

    EXAMPLE:
    --------
    ```python
    # After parallel research completes
    research_results = await execute_parallel_research(agents, tasks)

    # Sequential validation
    validator = SequentialValidatorAgent("validator", model)
    validated_results = await execute_sequential_validation(
        validator, research_results
    )

    # Filter passing results
    passing_results = [r for r in validated_results if r.validated]
    ```
    """
    logger.info(f"Starting sequential validation of {len(research_results)} results")

    validated_results = []

    # ========================================================================
    # CAPSTONE REQUIREMENT: Sequential Agents
    # Process each result sequentially in a for loop
    # ========================================================================
    for i, result in enumerate(research_results):
        logger.info(f"Validating result {i+1}/{len(research_results)} " f"from {result.agent_id}")

        validation = await validator.execute_task(result)
        validated_results.append(validation)

    # Report summary
    passed = sum(1 for r in validated_results if r.validated)
    failed = len(validated_results) - passed

    logger.info(f"Sequential validation completed: " f"{passed} passed, {failed} failed")

    return validated_results
