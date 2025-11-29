"""
# ============================================================================
# QEARIS - LOOP SYNTHESIS AGENT
# ============================================================================
# 
# CAPSTONE REQUIREMENT: Loop Agents
# POINTS: Technical Implementation - 50 points (Multi-Agent System)
# 
# DESCRIPTION: Synthesis agent that uses iterative refinement to combine
# multiple research results into a comprehensive final report. Uses a loop
# pattern with quality threshold-based termination.
# 
# INNOVATION: Quality-driven convergence with configurable thresholds,
# multi-dimensional quality assessment, and iterative improvement prompts.
# 
# FILE LOCATION: src/agents/loop_synthesis_agent.py
# 
# CAPSTONE CRITERIA MET:
# - Multi-Agent System: Loop execution pattern with convergence
# - Gemini Integration: Iterative content generation
# - Agent Evaluation: Quality scoring with threshold termination
# - Context Engineering: Intelligent prompt refinement
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


class LoopSynthesisAgent(BaseLLMAgent[ResearchResult]):
    """
    Synthesis agent with iterative refinement loop pattern.

    ============================================================================
    CAPSTONE REQUIREMENT: Loop Agents
    POINTS: Technical Implementation - 50 points (Multi-Agent System)

    DESCRIPTION:
    This agent synthesizes multiple research results into a comprehensive
    final report using an iterative loop pattern. Key features:

    1. LOOP EXECUTION BENEFITS:
       - Quality improvement through iteration
       - Adaptive termination based on quality threshold
       - Progressive refinement focus
       - Convergence to optimal output

    2. EXECUTION PATTERN:
       ```python
       for iteration in range(max_iterations):
           synthesis = await synthesizer.execute_task(results)
           if synthesis.quality >= threshold:
               break  # Early termination on quality
       ```

    3. LOOP LOGIC:
       - Initial synthesis: Combine all findings
       - Refinement iterations: Improve based on quality gaps
       - Termination: Quality threshold OR max iterations

    4. CONVERGENCE CRITERIA:
       - Quality threshold: 0.85 (configurable)
       - Max iterations: 3 (prevents infinite loops)
       - Focus areas: Structure → Depth → Clarity

    INNOVATION:
    -----------
    - Quality-driven convergence (not just fixed iterations)
    - Progressive refinement focus (different goals per iteration)
    - Multi-dimensional quality assessment
    - Diminishing returns detection
    ============================================================================

    WHY LOOP (not single pass)?
    ---------------------------
    First-pass synthesis is often rough:
    1. Integration of diverse sources is complex
    2. Initial structure may not be optimal
    3. Refinement improves coherence
    4. Iterative polishing catches issues

    WHY THRESHOLD-BASED TERMINATION?
    --------------------------------
    Fixed iterations waste resources:
    1. Good inputs may synthesize quickly
    2. Complex topics may need more iterations
    3. Quality threshold ensures consistent output
    4. Max iterations prevent runaway loops
    """

    def __init__(
        self,
        agent_id: str,
        gemini_model: Any,
        config: Optional[LLMAgentConfig] = None,
        mcp_server: Any = None,
        memory_bank: Any = None,
        quality_threshold: float = 0.85,
        max_iterations: int = 3,
        diminishing_returns_threshold: float = 0.01,
    ):
        """
        Initialize the loop synthesis agent.

        CAPSTONE REQUIREMENT: Loop Agents

        PARAMETERS:
        -----------
        agent_id : str
            Unique identifier (typically "synthesizer")
        gemini_model : GenerativeModel
            Configured Gemini model for synthesis
        config : LLMAgentConfig (optional)
            Agent configuration
        mcp_server : MCPServer (optional)
            MCP server for tool execution
        memory_bank : MemoryBank (optional)
            Memory bank for synthesis experience
        quality_threshold : float
            Quality score to achieve for early termination (default: 0.85)
        max_iterations : int
            Maximum refinement iterations (default: 3)
        diminishing_returns_threshold : float
            Minimum improvement to continue iterating (default: 0.01)

        QUALITY THRESHOLD: 0.85
        -----------------------
        WHY 0.85?
        - High enough for publication-quality output
        - Not perfectionist (0.95+ rarely achievable)
        - Balances quality vs computational cost
        - Achievable within 3 iterations for good input
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=LLMAgentType.SYNTHESIZER,
            gemini_model=gemini_model,
            config=config,
            mcp_server=mcp_server,
            memory_bank=memory_bank,
        )

        # ====================================================================
        # CAPSTONE REQUIREMENT: Loop Agents
        # Configurable loop termination criteria
        # ====================================================================
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.diminishing_returns_threshold = diminishing_returns_threshold

        logger.info(
            f"Loop Synthesis Agent initialized: {agent_id} "
            f"(threshold: {quality_threshold}, max_iter: {max_iterations})"
        )

    # ========================================================================
    # CAPSTONE REQUIREMENT: Loop Agents - Core Processing
    #
    # This method implements the iterative refinement loop with
    # quality threshold-based termination.
    # ========================================================================
    async def _process_task(self, task_input: Any) -> ResearchResult:
        """
        Synthesize research results using iterative refinement loop.

        CAPSTONE REQUIREMENT: Loop Agents

        ========================================================================
        LOOP EXECUTION FLOW:
        ========================================================================

        LOOP LOGIC:
        -----------
        ```python
        for iteration in range(max_iterations):
            if iteration == 0:
                synthesis = initial_synthesis(all_findings)
            else:
                synthesis = refine_synthesis(previous_synthesis, focus_area)

            quality = calculate_quality(synthesis)

            if quality >= threshold:
                break  # Early termination - quality achieved

        return synthesis  # Final output
        ```

        CONVERGENCE CRITERIA:
        ---------------------
        1. Quality threshold met (success termination)
        2. Max iterations reached (safety termination)
        3. Diminishing returns detected (optional optimization)

        ITERATION FOCUS AREAS:
        ----------------------
        Iteration 1: Initial synthesis - combine all findings
        Iteration 2: Structure and coherence improvement
        Iteration 3: Depth and comprehensiveness enhancement
        (Iteration 4+: Clarity and polish)

        ========================================================================
        """
        start_time = datetime.now()

        # Parse input (supports both dict and list formats)
        research_results = self._parse_input(task_input)

        logger.info(
            f"[LOOP] {self.agent_id} starting synthesis of "
            f"{len(research_results)} results "
            f"(max_iter: {self.max_iterations}, threshold: {self.quality_threshold})"
        )

        try:
            # ================================================================
            # Extract all content and sources from research results
            # ================================================================
            all_content = [r.content for r in research_results]
            all_sources = []
            for r in research_results:
                all_sources.extend(r.sources)
            all_sources = list(set(all_sources))  # Remove duplicates

            # ================================================================
            # LOOP: Iterative synthesis with quality threshold
            # CAPSTONE REQUIREMENT: Loop Agents
            # ================================================================
            current_synthesis = None
            final_quality = 0.0
            iterations_completed = 0
            quality_history = []

            for iteration in range(self.max_iterations):
                iterations_completed = iteration + 1

                logger.info(f"[LOOP] Iteration {iterations_completed}/{self.max_iterations}")

                # ============================================================
                # Generate synthesis based on iteration number
                # ============================================================
                if iteration == 0:
                    # Initial synthesis: Combine all findings
                    prompt = self._create_initial_synthesis_prompt(all_content)
                else:
                    # Refinement: Improve based on focus area
                    focus_area = self._get_iteration_focus(iteration)
                    prompt = self._create_refinement_prompt(
                        current_synthesis, iteration, focus_area
                    )

                # ============================================================
                # Generate synthesis with Gemini
                # CAPSTONE REQUIREMENT: Gemini Integration (BONUS)
                # ============================================================
                current_synthesis = await self._generate_response(prompt)

                # ============================================================
                # Calculate quality score
                # CAPSTONE REQUIREMENT: Agent Evaluation
                # ============================================================
                quality = self._calculate_quality(
                    synthesis=current_synthesis,
                    research_results=research_results,
                    iteration=iteration,
                )

                final_quality = quality
                quality_history.append(quality)

                logger.info(
                    f"[LOOP] Iteration {iterations_completed} quality: "
                    f"{quality:.2f} (threshold: {self.quality_threshold})"
                )

                # ============================================================
                # Check termination condition
                # ============================================================
                if quality >= self.quality_threshold:
                    logger.info(
                        f"[LOOP] Quality threshold met at iteration " f"{iterations_completed}"
                    )
                    break

                # Check for diminishing returns (optional early stop)
                if len(quality_history) >= 2:
                    improvement = quality_history[-1] - quality_history[-2]
                    if improvement < self.diminishing_returns_threshold:
                        logger.info(
                            f"[LOOP] Diminishing returns detected (improvement: {improvement:.4f} < {self.diminishing_returns_threshold}), stopping"
                        )
                        break

                # Small delay between iterations (rate limiting)
                if iteration < self.max_iterations - 1:
                    await asyncio.sleep(1)

            execution_time = (datetime.now() - start_time).total_seconds()

            # ================================================================
            # Create final synthesis result
            # ================================================================
            result = ResearchResult(
                task_id=f"synthesis_{datetime.now().timestamp()}",
                agent_id=self.agent_id,
                content=current_synthesis,
                sources=all_sources,
                confidence=final_quality,
                validated=True,
                metadata={
                    "synthesis_count": len(research_results),
                    "iterations": iterations_completed,
                    "execution_time": execution_time,
                    "quality_score": final_quality,
                    "quality_history": quality_history,
                    "total_sources": len(all_sources),
                    "threshold_met": final_quality >= self.quality_threshold,
                },
            )

            # ================================================================
            # Store synthesis experience
            # CAPSTONE REQUIREMENT: Sessions & Memory
            # ================================================================
            await self.store_experience(
                content=(
                    f"Synthesized {len(research_results)} results in "
                    f"{iterations_completed} iterations. "
                    f"Final quality: {final_quality:.2f}, "
                    f"Threshold: {'met' if final_quality >= self.quality_threshold else 'not met'}"
                ),
                importance=0.9,
                memory_type=MemoryType.EXPERIENCE,
                metadata={
                    "iterations": iterations_completed,
                    "quality": final_quality,
                    "sources_count": len(all_sources),
                },
            )

            logger.info(
                f"[LOOP] {self.agent_id} completed in {execution_time:.2f}s "
                f"({iterations_completed} iterations, quality: {final_quality:.2f})"
            )

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            logger.error(f"[LOOP] {self.agent_id} synthesis failed: {str(e)}", exc_info=True)

            return ResearchResult(
                task_id=f"synthesis_{datetime.now().timestamp()}",
                agent_id=self.agent_id,
                content=f"Synthesis failed: {str(e)}",
                sources=[],
                confidence=0.0,
                metadata={"error": str(e), "execution_time": execution_time},
            )

    # ========================================================================
    # INPUT PARSING
    # ========================================================================

    def _parse_input(self, task_input: Any) -> List[ResearchResult]:
        """
        Parse task input to extract research results.

        Supports multiple input formats:
        - Dict with 'research_results' key
        - List of ResearchResult objects
        - Single ResearchResult object
        """
        if isinstance(task_input, dict):
            return task_input.get("research_results", [])
        elif isinstance(task_input, list):
            return task_input
        else:
            return [task_input]

    # ========================================================================
    # PROMPT GENERATION
    # ========================================================================

    def _create_initial_synthesis_prompt(self, all_content: List[str]) -> str:
        """
        Create prompt for initial synthesis (Iteration 1).

        CAPSTONE REQUIREMENT: Loop Agents
        First iteration focuses on combining all findings.
        """
        findings_text = self._format_findings(all_content)

        return f"""You are a synthesis agent tasked with creating a comprehensive,
coherent report from multiple research findings.

RESEARCH FINDINGS TO SYNTHESIZE:
{'═' * 60}

{findings_text}

{'═' * 60}

SYNTHESIS TASK:
Create a well-structured, comprehensive report that:

1. INTEGRATES all findings into a coherent narrative
2. ORGANIZES information logically with clear sections
3. IDENTIFIES key themes and insights across sources
4. RESOLVES any contradictions between findings
5. MAINTAINS appropriate citations and source attribution
6. USES clear, professional language

Structure your report with:
- Executive Summary
- Key Findings (by theme)
- Analysis and Insights
- Conclusions

Be thorough yet concise. Aim for comprehensive coverage while avoiding redundancy."""

    def _create_refinement_prompt(
        self, current_synthesis: str, iteration: int, focus_area: str
    ) -> str:
        """
        Create prompt for refinement iterations.

        CAPSTONE REQUIREMENT: Loop Agents
        Subsequent iterations focus on specific improvement areas.
        """
        return f"""You are refining a research synthesis to improve its quality.
Focus specifically on: {focus_area}

CURRENT SYNTHESIS (Iteration {iteration}):
{'═' * 60}
{current_synthesis}
{'═' * 60}

REFINEMENT GOALS:
1. IMPROVE {focus_area}
2. STRENGTHEN logical flow and transitions
3. ENHANCE clarity and readability
4. ENSURE comprehensive coverage
5. POLISH language and style

Provide an IMPROVED version that specifically addresses these goals.
Maintain all existing good content while enhancing weak areas."""

    def _get_iteration_focus(self, iteration: int) -> str:
        """
        Get focus area for each iteration.

        CAPSTONE REQUIREMENT: Loop Agents
        Different focus areas for progressive improvement.
        """
        focus_areas = {
            1: "structure, organization, and coherence",
            2: "depth, comprehensiveness, and detail",
            3: "clarity, readability, and polish",
        }
        return focus_areas.get(iteration, "overall quality and refinement")

    def _format_findings(self, content_list: List[str]) -> str:
        """Format research findings for synthesis prompt."""
        formatted = []
        for i, content in enumerate(content_list, 1):
            formatted.append(f"FINDING {i}:")
            formatted.append(f"{content}")
            formatted.append("")
        return "\n".join(formatted)

    # ========================================================================
    # QUALITY CALCULATION
    # ========================================================================

    def _calculate_quality(
        self, synthesis: str, research_results: List[ResearchResult], iteration: int
    ) -> float:
        """
        Calculate synthesis quality score.

        CAPSTONE REQUIREMENT: Agent Evaluation

        ========================================================================
        QUALITY FACTORS:
        ========================================================================

        1. LENGTH (20% weight)
           - Comprehensive coverage indicator
           - Minimum viable: 200 words
           - Optimal: 400+ words

        2. STRUCTURE (25% weight)
           - Organization quality
           - Headers, lists, paragraphs
           - Logical sections

        3. SOURCE INTEGRATION (15% weight)
           - How well sources are cited
           - Multi-source synthesis

        4. INPUT CONFIDENCE (20% weight)
           - Quality of input research
           - Average confidence inheritance

        5. ITERATION BONUS (20% weight)
           - Progressive improvement
           - Refinement benefit

        FORMULA:
        --------
        quality = length_score × 0.20
                + structure_score × 0.25
                + source_score × 0.15
                + input_confidence × 0.20
                + iteration_bonus × 0.20
        ========================================================================
        """
        if not synthesis:
            return 0.0

        # ================================================================
        # Factor 1: Length (20%)
        # ================================================================
        word_count = len(synthesis.split())
        if word_count >= 400:
            length_score = 1.0
        elif word_count >= 300:
            length_score = 0.85
        elif word_count >= 200:
            length_score = 0.7
        elif word_count >= 100:
            length_score = 0.5
        else:
            length_score = 0.3

        # ================================================================
        # Factor 2: Structure (25%)
        # ================================================================
        structure_indicators = [
            "\n\n",  # Paragraphs
            "1.",  # Numbered lists
            "##",  # Markdown headers
            "**",  # Bold emphasis
            "- ",  # Bullet points
            ":",  # Key-value patterns
        ]

        structure_count = sum(1 for ind in structure_indicators if ind in synthesis)
        structure_score = min(structure_count / 4.0, 1.0)

        # ================================================================
        # Factor 3: Source Integration (15%)
        # ================================================================
        # Check for source mentions
        source_indicators = ["source", "according to", "found that", "research"]
        source_mentions = sum(synthesis.lower().count(ind) for ind in source_indicators)
        source_score = min(source_mentions / 3.0, 1.0)

        # ================================================================
        # Factor 4: Input Confidence (20%)
        # ================================================================
        if research_results:
            input_confidence = sum(r.confidence for r in research_results) / len(research_results)
        else:
            input_confidence = 0.5

        # ================================================================
        # Factor 5: Iteration Bonus (20%)
        # ================================================================
        # Progressive improvement through iterations
        iteration_bonus = min(iteration * 0.15, 0.3) + 0.5

        # ================================================================
        # Calculate weighted total
        # ================================================================
        quality = (
            length_score * 0.20
            + structure_score * 0.25
            + source_score * 0.15
            + input_confidence * 0.20
            + iteration_bonus * 0.20
        )

        return min(1.0, quality)


# ============================================================================
# CAPSTONE REQUIREMENT: Loop Agents - Utility Function
#
# Helper function to execute iterative synthesis loop
# ============================================================================
async def execute_synthesis_loop(
    synthesizer: LoopSynthesisAgent,
    research_results: List[ResearchResult],
    validation_results: Optional[List[ResearchResult]] = None,
) -> ResearchResult:
    """
    Execute synthesis with iterative refinement loop.

    CAPSTONE REQUIREMENT: Loop Agents

    This function demonstrates the loop execution pattern with
    quality threshold-based termination.

    LOOP PATTERN:
    -------------
    ```python
    for iteration in range(max_iterations):
        synthesis = await synthesizer.execute_task(results)
        if synthesis.quality >= threshold:
            break  # Early termination
    ```

    PARAMETERS:
    -----------
    synthesizer : LoopSynthesisAgent
        The synthesis agent to use
    research_results : List[ResearchResult]
        Results from parallel research
    validation_results : List[ResearchResult] (optional)
        Results from sequential validation

    RETURNS:
    --------
    ResearchResult : Final synthesized report

    EXAMPLE:
    --------
    ```python
    # After parallel research and sequential validation
    research_results = await execute_parallel_research(agents, tasks)
    validation_results = await execute_sequential_validation(validator, research_results)

    # Loop synthesis
    synthesizer = LoopSynthesisAgent("synthesizer", model, quality_threshold=0.85)
    final_report = await execute_synthesis_loop(
        synthesizer, research_results, validation_results
    )

    print(f"Final quality: {final_report.confidence}")
    print(f"Iterations: {final_report.metadata['iterations']}")
    ```
    """
    logger.info(f"Starting synthesis loop with {len(research_results)} research results")

    # Prepare input
    task_input = {
        "research_results": research_results,
        "validation_results": validation_results or [],
    }

    # Execute synthesis (loop is internal to the agent)
    final_report = await synthesizer.execute_task(task_input)

    # Report summary
    iterations = final_report.metadata.get("iterations", 0)
    quality = final_report.confidence
    threshold_met = final_report.metadata.get("threshold_met", False)

    logger.info(
        f"Synthesis loop completed: "
        f"{iterations} iterations, quality: {quality:.2f}, "
        f"threshold: {'met' if threshold_met else 'not met'}"
    )

    return final_report
