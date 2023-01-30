# Aleatoric and Epistemic Uncertainty

In my past readings on urban driving, I have recently came lot more across on the topic of uncertainty, rather than purely observing the accuracy or other supervised metrics when discussing model performance and generalization for autonomous driving. As I was not so familiar with the topic in need of a recap, this articles tries to explain the two ideas, and link to some relevant works in self-driving academia.

*Expected reading time: 20 min*

**References**
1. <a href="https://www.gdsd.statistik.uni-muenchen.de/2021/gdsd_huellermeier.pdf">Eyke HullerMeier's presentation on Uncertainty in ML</a>
2. <a href="https://waymo.com/research/improving-the-intra-class-long-tail-in-3d-detection-via-rare-example-mining/">Improving the Intra-class Long-tail in 3D Detection via Rare Example Mining</a>
3. <a href="https://arxiv.org/pdf/1703.04977.pdf">What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?</a>
4. <a href="https://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf">Y.Gal, Uncertainty in Deep Learning</a>

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

### Bayesian DL Framework
Uncertainty in DL is either captured as a distribution over network's weights or over its outputs, where epistemic uncertainty is measured with prior distribution over weights and aleatoric uncertainty is captured by distribution over input [4].

As usual for Bayesian methodology, capturing model uncertainty involves setting a prior distribution over the model weights e.g. W~N(0,1). This type of network is referred to as Bayesian Neural Networks (BNNs). Model output from BNN then captures the expected value of model output w.r.t weights distribution. Bayesian inference allows evaluating P(W|X,Y), finding likelihood of weights W given the input learning dataset (X,Y).

A problem with BNN and Bayesian Inference is that the output likelihood p(y|x) cannot be evaluated anlytically (how does one marginalize this oveer all possible values of W?) Hence, it is instead approximated in various ways using a simpler distribution over possible network parameters. Dropout variational inference is one such approach, where dropout, typically used in training only as a regularization measure, is also used during inference to approximate variation of possible model parameters. This MC dropout is equivalent to running inference under the approximation of a model posterior P(W|X,Y) after the model has been trained.

The above MC dropout method thus makes measurement of epistemic uncertainty for DL models possible. The epistemic uncertainty is captured as either the entropy of the marginalized model output or the variance of model predictions. As models are introduced to more data, their uncertainty for either classification or regression would reduce to 0.

Meanwhile, heteroscedastic Aleatoric Uncertainty assumes variation of observation noise depending on the input x. This helps quantifying parts of the input where it may be more difficult for a model to predict e.g. identify inherently difficult scenes and/or objects. In this case MAP inference is used, meaning a single value is found for model parameters theta that approximates variance of model input i.e. heteroscedastic aleatoric uncertainty. The learned model should be such that if a model prediction is very off, the scaled variance should be quite large to represent the high uncertainty, whereas low for accurate model predictions. The model thus captures how varied the model error is given the input.

### Combining Heteroscedastic Aleatoric Uncertainty and Epistemic Uncertainty.

In order to combine the two, the learned NN model for heteroscedastic aleatoric uncertainty has to be considered as BNN, similar to the construction of capturing epistemic uncertainty, such that they are uniformly constructed in the same model. What does this mean? Let us consider this with code for concrete understanding.


Therefore, with the refactoring of the aleatoric model, we are now predicting combined model uncertainty and input uncertainty by running bayesian inference to get not only varying samples of model output but also the corresponding aleatoric uncertainty, hence providing variance on not only the model output but also the input.

### Loss attenuation with Heteroscedastic Uncertainty
Given the formal definition and construction of uncertainty, as well as how to capture them, one question that may arise is how they can be used to further robustify DL. One approach is to consider them in learning, where uncertainty is used to attenuate model loss accordingly. 