# Better RAG using Relevant Information Gain

[[ArXiv]](https://arxiv.org/abs/2407.12101)

## Background

Retrieval Augmented Generation, or RAG, combines a retrieval engine with a generative model i.e. LLM to provide relevant context as input to the model. While this allows the model to handle up-to-date information and does not require retraining to obtain new knowledge, this bears the question of optimal search.

A typical RAG uses either sparse retrieval e.g. TFIDF, BM25, or dense retrieval e.g. E5, OpenAI-text-embedding-3. More recently, cross encoder i.e reranker models have been considered to find more accurate and relevant context.

## The problem

However, authors of Better RAG mention the problem with aforementioned retrieval methods. All methods *return the highest individually relevant passage*, and therefore the retrieved context themselves do not consider redundancy.

## The solution

To this, authors propose combining the retrieval methods with *Dartboard* algorithm, "a principled retrieval method based on optimizing a simple metric of total information gain".

Suppose a two player game where player 1 selects a target T and gives query q, hoping to find T. Player 2 makes a set of guesses on what T may be, resulting in scores that evaluate liklelihood of correct guess:
$$s(G,q,A,\sigma) = \sum_{t\in A} P(T=t|q,\sigma) min_{g\in G} D(t|g)$$

Where $D$ is a distance function. The Dartboard algorithm aims to maximize the above score, and therefore tries to cover all ground i.e. reduce redundancy while making sure the correct context is retrieived based on $D$. $P$ is modeled e.g. using embedding model with cosine similarity, while $D$ can be inverse of similarity measured between contexts. 

## Limitations
* While authors show the algorithm improves NDCG and MMR, they only conducted on a single benchmark.
* Hyper parameter $\sigma$ controls how diverse the retrieved context should be, and not much investigation has been done on tuning.

## Final thoughts
Overall a neat insight. I haved observed redundacy of retrieved contexts causing confusion to the LLM before. The algorithm/principle is in its early stages and more experiments may help refine it. 

