---
layout: post
---

# Notes on "What We Learned from a Year of Building with LLMs (Part I)"
## *LLM 기반 솔루션 개발자들의 직접적인 경험 공유*

After reading [this blog](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/) I have summarized here what I found to be most useful.


### 1. Prompting Tips

**For n-shot prompting:**

I. For n-shot prompting, if n grows too small the model will overfit to use the examples always as the anchor. Authors mention n >= 5 is a good rule of thumb, and don't be afraid to go as high as 20+.

II. 

Examples should be representative of the expected input distribution. If you’re building a movie summarizer, include samples from different genres in roughly the proportion you expect to see in practice.

You don’t necessarily need to provide the full input-output pairs. In many cases, examples of desired outputs are sufficient.

If you are using an LLM that supports tool use, your n-shot examples should also use the tools you want the agent to use.

2. Be more specific with CoT prompting
e.g.: "First, list the key decisions, follow-up items, and associated owners in a sketchpad.
Then, check that the details in the sketchpad are factually consistent with the transcript.
Finally, synthesize the key points into a concise summary."

3.Use deterministic flows for agents for now, even though they can technically behave dynamically
Thus, the likelihood that an agent completes a multi-step task successfully decreases exponentially as the number of steps increases. As a result, teams building agents find it difficult to deploy reliable agents.

A promising approach is to have agent systems that produce deterministic plans which are then executed in a structured, reproducible way. 

some things to try for planning agentic workflows:
Some things to try

An explicit planning step, as tightly specified as possible. Consider having predefined plans to choose from (c.f. https://youtu.be/hGXhFa3gzBs?si=gNEGYzux6TuB1del).
Rewriting the original user prompts into agent prompts. Be careful, this process is lossy!
Agent behaviors as linear chains, DAGs, and State-Machines; different dependency and logic relationships can be more and less appropriate for different scales. Can you squeeze performance optimization out of different task architectures?
Planning validations; your planning can include instructions on how to evaluate the responses from other agents to make sure the final assembly works well together.
Prompt engineering with fixed upstream state—make sure your agent prompts are evaluated against a collection of variants of what may happen before.

4. increasing diversity with prompt, rather than temperature

if prompt input uses list of historically purchased goods, mix them up! shift the prompt phrasing to encourage more diverse behavior

5. Evaluation
* assertion based unit-case with sample input-outputs help specify CI for specific-enough LLM application.
* LLM-as-judge works better for relative/pairwise comparisons, accepting ties, and with bias controls such as positional(shift inputs around and run multiple times) and length(ensure compared responses have similar length).
* use the intern test as problem identifier:


"We like to use the following “intern test” when evaluating generations: If you took the exact input to the language model, including the context, and gave it to an average college student in the relevant major as a task, could they succeed? How long would it take?

If the answer is no because the LLM lacks the required knowledge, consider ways to enrich the context.

If the answer is no and we simply can’t improve the context to fix it, then we may have hit a task that’s too hard for contemporary LLMs.

If the answer is yes, but it would take a while, we can try to reduce the complexity of the task. Is it decomposable? Are there aspects of the task that can be made more templatized?

If the answer is yes, they would get it quickly, then it’s time to dig into the data. What’s the model doing wrong? Can we find a pattern of failures? Try asking the model to explain itself before or after it responds, to help you build a theory of mind."