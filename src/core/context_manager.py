"""
Context Manager - Intelligent context window management

PURPOSE: Optimize context usage within token limits
TECHNIQUES: Prioritization, compaction, summarization
GOAL: Maximize relevant information while respecting limits
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContextItem:
    """
    Single item in context

    DESIGN: Prioritized items with metadata
    WHY: Enable intelligent selection when context is full
    """

    content: str
    priority: float  # 0.0 to 1.0
    category: str  # 'system', 'memory', 'rag', 'conversation'
    token_count: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ContextManager:
    """
    Intelligent context window management

    PROBLEM:
    --------
    - LLMs have token limits (Gemini: 1M tokens)
    - Not all context is equally important
    - Need to fit most relevant info

    SOLUTION:
    ---------
    1. Categorize context items
    2. Assign priorities
    3. Select optimally when limit reached
    4. Compact/summarize when needed

    MATHEMATICS:
    ------------
    Optimization problem:
    maximize: Σ (priority_i × relevance_i)
    subject to: Σ tokens_i ≤ max_tokens

    Greedy solution (good enough):
    Sort by priority, take top items until limit
    """

    def __init__(self, max_tokens: int = 100000):
        """
        Initialize context manager

        PARAMETERS:
        -----------
        max_tokens: Maximum context window size

        WHY 100K default?
        - Conservative estimate for safety
        - Leaves room for output
        - Can be increased if needed
        """
        self.max_tokens = max_tokens
        self.items: List[ContextItem] = []
        self.total_tokens = 0

        logger.info(f"ContextManager initialized with {max_tokens} token limit")

    def add_item(
        self,
        content: str,
        priority: float = 0.5,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add item to context

        PROCESS:
        1. Estimate token count
        2. Check if fits
        3. Add or reject

        PRIORITY GUIDELINES:
        - 1.0: Critical system prompts
        - 0.8: High-relevance memories
        - 0.6: RAG results
        - 0.4: Conversation history
        - 0.2: Background info

        RETURNS:
        - True if added
        - False if rejected (full)
        """
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        token_count = len(content) // 4

        if self.total_tokens + token_count > self.max_tokens:
            logger.warning(
                f"Context full: Cannot add {token_count} tokens "
                f"(current: {self.total_tokens}/{self.max_tokens})"
            )
            return False

        item = ContextItem(
            content=content,
            priority=priority,
            category=category,
            token_count=token_count,
            metadata=metadata or {},
        )

        self.items.append(item)
        self.total_tokens += token_count

        logger.debug(
            f"Added context item: {category} " f"({token_count} tokens, priority={priority:.2f})"
        )

        return True

    def build_context(
        self, include_categories: Optional[List[str]] = None, max_items: Optional[int] = None
    ) -> str:
        """
        Build final context string

        PROCESS:
        1. Filter by category (if specified)
        2. Sort by priority (descending)
        3. Take top items
        4. Concatenate

        STRUCTURE:
        [CATEGORY 1]
        content...

        [CATEGORY 2]
        content...

        WHY this structure?
        - Clear separation
        - Easy to parse
        - Category-aware
        """
        # Filter items
        items = self.items
        if include_categories:
            items = [item for item in items if item.category in include_categories]

        # Sort by priority
        items = sorted(items, key=lambda x: x.priority, reverse=True)

        # Limit count if specified
        if max_items:
            items = items[:max_items]

        # Build context
        context_parts = []
        current_category = None

        for item in items:
            if item.category != current_category:
                context_parts.append(f"\n[{item.category.upper()}]\n")
                current_category = item.category

            context_parts.append(item.content)
            context_parts.append("\n")

        context = "".join(context_parts)

        logger.info(
            f"Built context: {len(items)} items, "
            f"{len(context)} chars, "
            f"{len(context)//4} estimated tokens"
        )

        return context

    def compact(self, target_reduction: float = 0.3):
        """
        Compact context by removing low-priority items

        STRATEGY:
        Remove bottom 30% by priority

        WHY 30%?
        - Significant reduction
        - Keeps important content
        - Can repeat if needed

        USE CASE:
        When context approaching limit
        """
        if not self.items:
            return

        # Sort by priority
        sorted_items = sorted(self.items, key=lambda x: x.priority)

        # Calculate how many to remove
        n_remove = int(len(self.items) * target_reduction)

        # Remove lowest priority items
        items_to_remove = sorted_items[:n_remove]

        for item in items_to_remove:
            self.items.remove(item)
            self.total_tokens -= item.token_count

        logger.info(
            f"Compacted context: Removed {n_remove} items, "
            f"freed {sum(i.token_count for i in items_to_remove)} tokens"
        )

    def summarize_category(self, category: str, summary_ratio: float = 0.3):
        """
        Summarize all items in a category

        PROCESS:
        1. Collect all items in category
        2. Concatenate content
        3. Generate summary (using LLM would be ideal)
        4. Replace with summary

        NOTE: This version uses simple truncation
              In production, use LLM summarization

        WHY summarize?
        - Preserve key info
        - Reduce tokens
        - Maintain coherence
        """
        # Get items in category
        category_items = [item for item in self.items if item.category == category]

        if not category_items:
            return

        # Concatenate
        full_content = "\n".join([item.content for item in category_items])

        # Simple summarization (truncate)
        # In production: Use LLM or extractive summarization
        target_length = int(len(full_content) * summary_ratio)
        summary = full_content[:target_length] + "..."

        # Calculate tokens saved
        original_tokens = sum(item.token_count for item in category_items)
        summary_tokens = len(summary) // 4
        tokens_saved = original_tokens - summary_tokens

        # Remove original items
        for item in category_items:
            self.items.remove(item)
            self.total_tokens -= item.token_count

        # Add summary
        self.add_item(
            content=summary,
            priority=max(item.priority for item in category_items),
            category=f"{category}_summary",
            metadata={"summarized": True, "original_items": len(category_items)},
        )

        logger.info(
            f"Summarized {category}: "
            f"{len(category_items)} items → 1 summary, "
            f"saved {tokens_saved} tokens"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get context statistics"""
        categories = {}
        for item in self.items:
            if item.category not in categories:
                categories[item.category] = {"count": 0, "tokens": 0}
            categories[item.category]["count"] += 1
            categories[item.category]["tokens"] += item.token_count

        return {
            "total_items": len(self.items),
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "utilization": self.total_tokens / self.max_tokens,
            "by_category": categories,
        }

    def clear(self):
        """Clear all context"""
        self.items = []
        self.total_tokens = 0
        logger.info("Context cleared")
