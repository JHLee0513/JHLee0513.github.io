MILE: Wayve's approach to vision-baed Imitation Learning for Urban Driving

*This is an article based on Wayve.ai's work: <a href='https://arxiv.org/pdf/2210.07729.pdf'>Model-Based Imitation Learning for Urban Driving</a>.*

<a href='https://www.wayve.ai/'>Wayve</a>, a self driving company in UK, develops autonomous vehicle technology that they refer to as AV2.0, aligns a lot with what I believe could be the next step towards achieving efficient, reliable autonomous driving for complex urban environments. In this article I highlight some of their remarkable work, especially focus on their latest work MILE, in relation to what has been going on in the field pertaining to BEV perception and Imitation Learning.

<b>Model-Based Imitation Learning for Urban Driving</b>

<img src=/>
![]("images/MILE.png?raw=true")
*Above figure illustrates the overall architecture of MILE. Source: <a href='https://arxiv.org/pdf/2210.07729.pdf'>arxiv paper</a>*

There are a few set of things happening in MILE's learning process, which I'll explain one by one. First of all, the goal of MILE is to estimate latent dynamics - the transition of world model's state encoding the world environment, and its transition given actions taken by the agent exploring it. 


<b>Related work: Background on BEV Perception</b>

With MILE's architecture, it is obvious to see that it's derived from <aa href='https://arxiv.org/abs/2104.10490'>FIERY</a>, their previous paper which focus on camera based BEV perception instead of the IL component. 

<b>Related work: Background on Imitation Learning</b>

Understandably, evaluation of learning methods for urban driving is difficult for multiple reasons. 1) real-world testing is not only logistically difficult and expensive to achieve, but scaling it to thousands of experiments to gain a general trust on the evalauationn is almost intractable, 2) experiments in exact scenarios are difficult to reproduce, and 3) there lie randomness and uncertainty difficult to remove for a proper experimentation in the real world in comparison to offline data or simulation. Hence, many works including MILE is trained and evaluated on the popular CARLA simulator, which provides a common ground for evaluation (aligning with the 'common task framework' for reproducible, controlled, experimentation).

With the help of the CARLA simulator, various methods have been tested on the simulation and progress has been made to show more effective driving in the sim-controlled environment the past few years, table below highlights most of them:

| Method/paper | Year | Performance |
| -----------  | ---- | ----------- |
| Header       |  | Title       |
| Paragraph    |  | Text        |

<b>Related work: modeling behavior prediction</b>

Notable works are Nvdia's <a href="/blogs/blogs.md">PredictionNet</a>

<b>What comes next?</b>

We likely will expect continous additions to the above table, with more methods and engineering towards safe, reliable, and efficient urban driving. Furthermore, with current directions in research more attention is expected for scalable vision and planning. In author's point of view, other directions that should (and may) see more researach is towards the long tail (as everyone likes to first point out in their AV talks) - unexpected obstacle detection and handling, out of distribution data handling, domain adaptation, and self supervised learning.

<b> Additional Materials </b>
If the above article was interesting, consider checking out wayve's paper for more details, or the following list of items for further exploration:

CVPR WAD workshop: https://cvpr2022.wad.vision/ - many great speakers and challenges are presented.
My article on BEV perception (WIP)
