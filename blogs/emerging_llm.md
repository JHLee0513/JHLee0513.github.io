---
layout: post
---

# Emerging Architectures for LLM applications

Summary of rising methods for LLM applications.

## References:

1. Blog: https://a16z.com/emerging-architectures-for-llm-applications/
2. Video: https://www.youtube.com/watch?v=Pft04KLw5Lk&t=289s

## Retrival Augmented Generation

* LLMs do not have access to information outside of its training set, and therefore has to extrapolate for information it does not know
* Alleviate this limitation by augmenting it with a database to retrieve knowledge to answer prompts
* Since an external database is used, the LLM can be given latest information with flexible control
* Systemwise, this means we need LLM and a retrieval system for the LLM to interact with
* Common methods to retrieval system: embedding model and vector database to encode input query and retrieve relevant information
* LLMs then have to process the searched documents, input prompt as a form of in-context learning to provide output
* Vector DBs tend to be the 'hot' current method, but it's important to note that is a lot of ways to organize and search documents

### Limitation of LLMs
* LLM performance scales inversely with prompt size
* LLMs tend to select first option when provided multiple choice (selection bias), or it will prioritize the last task it is given (recency bias)