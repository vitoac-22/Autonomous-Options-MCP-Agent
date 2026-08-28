# Autonomous Options MCP Agent

## Overview
An institutional-grade Options Alpha Agent engineered for the Alpaca AI Trading Hackathon. This system discards static moving averages in favor of stochastic volatility modeling (GARCH) and executes multi-leg derivatives strategies autonomously.

## Core Architecture (SOLID Principles)
1. **Volatility Engine (SRP):** Ingests spot market log returns and projects conditional variance ($\sigma_t^2$) using a Skewed-t GARCH(1,1) model. 
2. **Derivatives Pricing:** Calculates theoretical options premiums via the Black-Scholes-Merton differential equation.
3. **Convex Optimization (OCP):** Resolves Delta-neutral contract sizing using strict integer programming (CVXPY) to ensure indivisible contract constraints.
4. **Cognitive LLM Routing (DIP):** Decouples vendor lock-in by utilizing Featherless AI endpoints[cite: 1]. The `Qwen/Qwen2.5-7B-Instruct` model parses portfolio state and structures trade logic.

## Transactional Integrity (ACID)
Multi-leg options trading (e.g., Iron Condors, Straddles) introduces asymmetric risk if partially filled. The agent enforces strict transactional atomicity:
* **Validation:** Contract sizing parameters are hard-casted to integers. Nominal values are casted to strings to satisfy the Alpaca MCP server schema.
* **Rollback:** If a single leg of a multi-leg strategy fails due to liquidity constraints, the execution block is aborted, and previous legs are immediately liquidated to maintain market neutrality.

## Infrastructure
* **API:** Alpaca Paper Trading (Options enabled).
* **LLM Engine:** Featherless AI (`Qwen/Qwen2.5-7B-Instruct`).
* **Protocol:** Model Context Protocol (MCP) via `uvx alpaca-mcp-server`.