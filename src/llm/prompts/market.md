# Market Intelligence Analysis Prompt

System: You analyze multiple search snippets about hiring requirements.

Given these snippets for {role} in {region}, extract:
- in_demand_skills [], common_tools [], frameworks [], nice_to_have [].

Normalize synonyms where obvious. Return tight JSON with arrays only.

## Output Format

```json
{
  "in_demand_skills": ["python", "kubernetes", "machine learning", "sql"],
  "common_tools": ["aws", "docker", "git", "jenkins"],
  "frameworks": ["tensorflow", "react", "django", "pytorch"],
  "nice_to_have": ["golang", "terraform", "graphql", "kafka"]
}
