# Shreya Krishna Capstone 3 README

The evidence for the various steps are in the corresponding .txt files. 

I was having some trouble with the lambda deployment stage. 

# Reflection

Question 6 most tested the orchestrator because it required routing to both RDS and document retrieval, then synthesizing results without attributing document content to SQL. The main failure was initially a synthesis problem, with some SQL-generation contamination; routing itself worked. This shows that multi-agent systems remain fragile at the handoff between agents—especially source attribution and unsupported claims—whereas a single well-scoped agent has a narrower context and fewer coordination points.

