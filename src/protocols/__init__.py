"""
# ============================================================================
# QEARIS Protocols Module
# ============================================================================
# 
# CAPSTONE REQUIREMENT: A2A Protocol
# POINTS: Technical Implementation - Part of 50 points
# 
# This module provides agent communication protocols:
# 
# COMPONENTS:
# - A2A Protocol: Agent-to-agent message passing
# ============================================================================
"""

from src.protocols.a2a_protocol import (
    A2AMessage,
    A2AMessageType,
    A2AProtocol,
    A2AResponse,
    create_a2a_protocol,
)

__all__ = [
    "A2AProtocol",
    "A2AMessage",
    "A2AMessageType",
    "A2AResponse",
    "create_a2a_protocol",
]
