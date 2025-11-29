#!/usr/bin/env python3
"""
# ============================================================================
# QEARIS - Simple Research Example
# ============================================================================
# 
# CAPSTONE REQUIREMENT: Examples and Documentation
# 
# DESCRIPTION: Simple example demonstrating basic QEARIS usage for
# research tasks. Shows how to:
# - Initialize the system
# - Execute a simple research query
# - Process results
# 
# USAGE:
#   python examples/simple_research.py
# 
# REQUIREMENTS:
#   - GEMINI_API_KEY environment variable set
#   - Dependencies installed (pip install -r requirements.txt)
# ============================================================================
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
import google.generativeai as genai


async def main():
    """
    Simple research example.
    
    This example demonstrates the basic workflow:
    1. Initialize Gemini model
    2. Create research query
    3. Execute research
    4. Display results
    """
    print("=" * 60)
    print("QEARIS - Simple Research Example")
    print("=" * 60)
    print()
    
    # ========================================================================
    # Step 1: Initialize Gemini
    # ========================================================================
    print("[Step 1] Initializing Gemini...")
    
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        print(f"  ✓ Gemini model: {settings.GEMINI_MODEL}")
    except Exception as e:
        print(f"  ✗ Failed to initialize Gemini: {e}")
        print("  Make sure GEMINI_API_KEY is set in your environment")
        return
    
    # ========================================================================
    # Step 2: Import QEARIS components
    # ========================================================================
    print("\n[Step 2] Loading QEARIS components...")
    
    from src.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator
    
    orchestrator = MultiAgentOrchestrator(model)
    print("  ✓ Multi-Agent Orchestrator initialized")
    
    # ========================================================================
    # Step 3: Define research query
    # ========================================================================
    print("\n[Step 3] Setting up research query...")
    
    query = "What are the key principles of quantum computing?"
    domains = ["quantum", "computer-science"]
    max_agents = 2
    
    print(f"  Query: {query}")
    print(f"  Domains: {domains}")
    print(f"  Max Agents: {max_agents}")
    
    # ========================================================================
    # Step 4: Execute research
    # ========================================================================
    print("\n[Step 4] Executing research...")
    print("  (This may take 30-60 seconds)")
    print()
    
    try:
        result = await orchestrator.research(
            query=query,
            domains=domains,
            max_agents=max_agents
        )
        
        # ====================================================================
        # Step 5: Display results
        # ====================================================================
        print("\n" + "=" * 60)
        print("RESEARCH RESULTS")
        print("=" * 60)
        print()
        
        print(f"Session ID:     {result['session_id']}")
        print(f"Status:         {result['status']}")
        print(f"Confidence:     {result['confidence']:.2f}")
        print(f"Sources:        {result['sources']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print()
        
        print("Metrics:")
        for key, value in result.get('metrics', {}).items():
            print(f"  {key}: {value}")
        print()
        
        print("Research Report:")
        print("-" * 60)
        # Print first 1000 characters
        report = result['result']
        if len(report) > 1000:
            print(report[:1000] + "...")
            print(f"\n[Report truncated - {len(report)} total characters]")
        else:
            print(report)
        
        print()
        print("=" * 60)
        print("Research complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"  ✗ Research failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
