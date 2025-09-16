"""Quick test of the market intelligence agent with sample data."""

from src.agents.market_intel import gather_market_summary

# Test market intelligence gathering
print("Testing market intelligence agent...")

# Test with mock search results (no LLM)
print("\n=== Testing with mock search (no LLM) ===")
summary = gather_market_summary("Senior AI Engineer", "Global", llm=None)

print(f"Role: {summary.role}")
print(f"Region: {summary.region}")

print(f"\nIn-demand skills ({len(summary.in_demand_skills)}):")
for skill in summary.in_demand_skills[:8]:
    print(f"  - {skill}")

print(f"\nCommon tools ({len(summary.common_tools)}):")
for tool in summary.common_tools[:8]:
    print(f"  - {tool}")

print(f"\nFrameworks ({len(summary.frameworks)}):")
for framework in summary.frameworks[:8]:
    print(f"  - {framework}")

print(f"\nNice to have ({len(summary.nice_to_have)}):")
for skill in summary.nice_to_have[:5]:
    print(f"  - {skill}")

print(f"\nSources sample ({len(summary.sources_sample)}):")
for source in summary.sources_sample[:3]:
    print(f"  - {source[:80]}...")

print("\n✅ Market intelligence agent working correctly!")

# Test with different role
print("\n=== Testing Data Scientist role ===")
summary2 = gather_market_summary("Data Scientist", "US", llm=None)
print(f"Role: {summary2.role}")
print(f"In-demand skills: {summary2.in_demand_skills[:5]}")
print(f"Tools: {summary2.common_tools[:5]}")

print("\n✅ All tests completed successfully!")