# Aleatoric and Epistemic Uncertainty

In my past readings on urban driving, I have recently came lot more across on the topic of uncertainty, rather than purely observing the accuracy or other supervised metrics when discussing model performance and generalization for autonomous driving. As I was not so familiar with the topic in need of a recap, this articles tries to explain the two ideas, and link to some relevant works in self-driving academia.

*10 min read*

**References**
1. <a href="https://www.gdsd.statistik.uni-muenchen.de/2021/gdsd_huellermeier.pdf">Eyke HullerMeier's presentation on Uncertainty in ML</a>
2. <a href="https://waymo.com/research/improving-the-intra-class-long-tail-in-3d-detection-via-rare-example-mining/">Improving the Intra-class Long-tail in 3D Detection via Rare Example Mining</a>
3. <a href="https://arxiv.org/pdf/1703.04977.pdf">What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?</a>

## Motivation

For pretty much a decade now, Machine Learning coupled with growth in data observed remarkable performance improvements in various fields, including but not limited to robotics, vision, and healthcare. However, as these systems' deployment are more considered and adopted to fields such as driving and healthcare where transparency, interpretability, and safety are emphasized, it becomes ever so more important to properly understands where AI fails and how certain it is of its answers.

An interesting example is the Stack Overflow's temporary ban of ChatGPT in its Q&A platform for programmers (<a href="https://meta.stackoverflow.com/questions/421831/temporary-policy-chatgpt-is-banned">link</a>). As the moderators point out, *"because the average rate of getting correct answers from ChatGPT is too low, the posting of answers created by ChatGPT is substantially harmful to the site and to users who are asking or looking for correct answers."* Furthermore, the largest issue with the wrong answers is that *"the answers which ChatGPT produces have a high rate of being incorrect, they typically look like they might be good and the answers are very easy to produce."* This hinders the proper integration of AI assisted systems into interactions with humans, and it's also where uncertainty can hopefull more easily and safely allow the users and sub systems to handle the AI generated output.

In this blog I go over the two main concepts on uncertainty used in ML/DL, specifically *epistemic* and *aleatoric* uncertainty, with references from various sources.

## Aleatoric, Epistemic?

$$ Total Uncertainty = Aleatoric Uncertainty + Epistemic Uncertainty $$

Aleatoric (statistical) uncertainty captures randomness inherent to the problem at hand, essentially consider the "variability in the outcome of an experiment which is due to inherently random effects." Meanwhile, epistemic (systematic) uncertainty refers to "lack of knowledge", where the uncertainty may be derived from lack of data or lack of understanding by the AI agent [1]. This provides an interesting point that while both types of uncertainty are important to downstream users/systems, epistemic uncertainty seems be the only type of uncertainty that can be further reduced during the learning process on a AI agent i.e. mine more challenging or uncertainty-inducing examples, build more generalizable models, etc. An application of this idea is observed in Waymo's work on hard example mining [2].

Further breakdown on each type of uncertainty are below.

### Epistemic Uncertainty

### Aleatoric Uncertainty

## Measuring Uncertainty in the Context of Deep Learning

To better understand how the two types of uncertainties are measured and used in the context of Deep Learning based perception, we refer to Alex Kendall et al.'s work [3]. 

Another formulation to refer to aleatoric and epsitemic uncertainty are homoscedastic and heteroscedastic uncertainty i.e. aleatoric uncertainty is inherently constantly present (does not change based on data or input), while for heteroscedastic unceratinty it changes as the model is introduced to more data.

The bayesian framework proposed to map model input to aleatoric and epsitemic uncertainty[3] allows to distinguish what inputs are genuinely difficult and hard to capture from those that could be improved with more data. 




