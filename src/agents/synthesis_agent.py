"""
Synthesis Agent - Iterative refinement loop

SPECIALIZATION: Multi-source integration
PATTERN: Loop execution
GOAL: Coherent, comprehensive final report
"""

import asyncio
from datetime import datetime
from typing import List
import logging

from src.agents.base_agent import BaseAgent
from src.orchestrator.task_models import ResearchResult

logger = logging.getLogger(__name__)


class SynthesisAgent(BaseAgent):
    """
    Synthesis agent for iterative report generation
    
    ROLE IN SYSTEM:
    ---------------
    Final stage that combines multiple research results
    into a coherent, comprehensive report using iterative
    refinement.
    
    LOOP PROCESS:
    -------------
    1. Combine all findings
    2. Check quality threshold (0.85)
    3. If below: Refine and repeat (max 3 iterations)
    4. If above: Accept and return
    
    WHY iterative?
    - First draft often rough
    - Refinement improves coherence
    - Diminishing returns after 3 iterations
    """
    
    async def execute_task(
        self,
        research_results: List[ResearchResult],
        max_iterations: int = 3
    ) -> ResearchResult:
        """
        Synthesize multiple results into final report
        
        MOTIVATION:
        -----------
        Multiple research agents produce domain-specific
        results. Need to combine into single, coherent
        report that:
        - Integrates all findings
        - Resolves contradictions
        - Maintains logical flow
        - Cites sources appropriately
        
        QUALITY THRESHOLD: 0.85
        WHY 0.85?
        - High enough for quality
        - Not perfectionist (0.95+ rarely achievable)
        - Balances quality vs time
        """
        start_time = datetime.now()
        
        logger.info(
            f"{self.agent.name} synthesizing {len(research_results)} results"
        )
        
        try:
            # Collect all content and sources
            all_content = [r.content for r in research_results]
            all_sources = []
            for r in research_results:
                all_sources.extend(r.sources)
            
            # Remove duplicates
            all_sources = list(set(all_sources))
            
            # Iterative synthesis loop
            current_synthesis = None
            quality_threshold = 0.85
            final_quality = 0.0
            
            for iteration in range(max_iterations):
                logger.info(
                    f"Synthesis iteration {iteration + 1}/{max_iterations}"
                )
                
                # Create prompt for this iteration
                if iteration == 0:
                    prompt = self._create_initial_prompt(all_content)
                else:
                    prompt = self._create_refinement_prompt(
                        current_synthesis,
                        iteration
                    )
                
                # Generate synthesis with Gemini
                response = await asyncio.to_thread(
                    self.gemini_model.generate_content,
                    prompt
                )
                
                current_synthesis = response.text
                
                # Calculate quality score
                quality = self._calculate_quality(
                    current_synthesis,
                    research_results,
                    iteration
                )
                
                final_quality = quality
                
                logger.info(
                    f"Quality score: {quality:.2f} "
                    f"(threshold: {quality_threshold})"
                )
                
                # Check if quality threshold met
                if quality >= quality_threshold:
                    logger.info(
                        f"Quality threshold met at iteration {iteration + 1}"
                    )
                    break
                
                # Add small delay between iterations
                if iteration < max_iterations - 1:
                    await asyncio.sleep(1)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create final result
            result = ResearchResult(
                task_id=f"synthesis_{datetime.now().timestamp()}",
                agent_id=self.agent.id,
                content=current_synthesis,
                sources=all_sources,
                confidence=final_quality,
                validated=True,
                metadata={
                    'synthesis_count': len(research_results),
                    'iterations': iteration + 1,
                    'execution_time': execution_time,
                    'quality_score': final_quality,
                    'total_sources': len(all_sources)
                }
            )
            
            # Store synthesis experience
            await self.store_experience(
                f"Synthesized {len(research_results)} results "
                f"in {iteration + 1} iterations. "
                f"Final quality: {final_quality:.2f}",
                importance=0.9
            )
            
            self.update_metrics(execution_time, success=True)
            
            logger.info(
                f"{self.agent.name} completed in {execution_time:.2f}s "
                f"({iteration + 1} iterations, quality: {final_quality:.2f})"
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"{self.agent.name} failed: {str(e)}", exc_info=True)
            
            self.update_metrics(execution_time, success=False)
            
            return ResearchResult(
                task_id=f"synthesis_{datetime.now().timestamp()}",
                agent_id=self.agent.id,
                content=f"Synthesis failed: {str(e)}",
                sources=[],
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _create_initial_prompt(self, all_content: List[str]) -> str:
        """Create prompt for initial synthesis"""
        return f"""You are a synthesis agent tasked with creating a comprehensive, 
coherent report from multiple research findings.

RESEARCH FINDINGS:
{'═' * 60}

{self._format_findings(all_content)}

{'═' * 60}

TASK:
Create a well-structured, comprehensive report that:
1. Integrates all findings into a coherent narrative
2. Organizes information logically
3. Identifies key themes and insights
4. Resolves any contradictions
5. Maintains appropriate citations
6. Uses clear, professional language

Provide a synthesis that is thorough yet concise."""
    
    def _create_refinement_prompt(self, current: str, iteration: int) -> str:
        """Create prompt for refinement iterations"""
        focus_areas = {
            1: "structure and coherence",
            2: "depth and comprehensiveness",
            3: "clarity and polish"
        }
        
        focus = focus_areas.get(iteration, "overall quality")
        
        return f"""Refine this synthesis to improve {focus}:

CURRENT SYNTHESIS:
{'═' * 60}
{current}
{'═' * 60}

REFINEMENT GOALS (Iteration {iteration}):
1. Improve {focus}
2. Strengthen logical flow
3. Enhance clarity and readability
4. Ensure comprehensive coverage
5. Polish language and style

Provide an improved version that addresses these goals."""
    
    def _format_findings(self, content_list: List[str]) -> str:
        """Format research findings for prompt"""
        formatted = []
        for i, content in enumerate(content_list, 1):
            formatted.append(f"Finding {i}:")
            formatted.append(f"{content}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _calculate_quality(
        self,
        synthesis: str,
        research_results: List[ResearchResult],
        iteration: int
    ) -> float:
        """
        Calculate synthesis quality
        
        FACTORS:
        --------
        1. Length (comprehensive?)
        2. Structure (organized?)
        3. Source integration
        4. Input confidence
        5. Iteration bonus
        
        FORMULA:
        --------
        base = 0.5
        + 0.2 × length_factor
        + 0.2 × structure_factor
        + 0.1 × avg_input_confidence
        + 0.05 × iteration (improvement bonus)
        """
        quality = 0.5
        
        # Factor 1: Length
        word_count = len(synthesis.split())
        if word_count > 300:
            length_factor = 0.2
        elif word_count > 200:
            length_factor = 0.15
        elif word_count > 100:
            length_factor = 0.1
        else:
            length_factor = 0.05
        
        quality += length_factor
        
        # Factor 2: Structure
        structure_indicators = [
            '\n\n',      # Paragraphs
            '1.',        # Numbered lists
            '##',        # Headers
            '- ',        # Bullets
        ]
        
        structure_score = sum(
            1 for ind in structure_indicators
            if ind in synthesis
        ) / len(structure_indicators)
        
        quality += structure_score * 0.2
        
        # Factor 3: Input confidence
        if research_results:
            avg_confidence = sum(
                r.confidence for r in research_results
            ) / len(research_results)
            quality += avg_confidence * 0.1
        
        # Factor 4: Iteration bonus (improvement over iterations)
        quality += min(iteration * 0.05, 0.15)
        
        return min(1.0, quality)
