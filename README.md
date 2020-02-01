# BeyondLDA
1. Numpy-based PLSA (Probabilistic Latent Semantic Analysis), **plsa.py**.
2. LDA (Latent Dirichlet Allocation) vs hLDA (Hierarchical Latent Dirichlet Allocation) and vizualization **LDA_vs.hLDA.ipynb**

## PLSA(plsa.py)
**All numpy-based implementation in PLSA**<br/><br/>
This was when I took Text Mining lecture by UIUC MSCS Dept. _Prof.ChengXiang Zhai_, where we are asked to build the [PLSA](https://arxiv.org/pdf/1301.6705.pdf) model in numpy.
PLSA learns both document-topic distribution and topic-word distribution by Bayes inference and optimize the log-likelihood by EM algorithm. The code comments in **plsa.py** includes
* How to initialized the parameters
* How to do E-step
* How to do M-step
* How to calculate the log-likelihood of the inference


## LDA and hLDA(LDA_vs_hLDA.ipynb)
**Summary and visual comparison between LDA and hLDA**<br/><br/>
This was when I took one of the topic mining campus analytic challenges in the U.S, where we're asked to provide sth more than [LDA](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf).<br/> The corpus used here is from NASA dataset with some masked token to make the task harder.<br/><br/>![Word Cloud](/Image/wordcloud.png)<br/><br/><br/>
This one starts from using [gensim API](https://radimrehurek.com/gensim/models/ldamodel.html) to lemmatize and add bi-gram token, and then train LDA model and visualize by [pyLDAvis](https://github.com/bmabey/pyLDAvis) and IPython widget for interactive result.<br/><br/> ![pyLDAvis](/Image/pyLDAvis.png) <br/><br/><br/>
For currently there is no popular Python packages with implementation on [hLDA](https://papers.nips.cc/paper/2466-hierarchical-topic-models-and-the-nested-chinese-restaurant-process.pdf). The file **hlda_sampler.py** is referred to [joewand's github](https://github.com/joewandy/hlda/blob/master/hlda/sampler.py). hlda_sampler.py is the Gibbs sampler for hLDA inference, based on the implementation from Mallet having a fixed depth on the nCRP tree. The most distinguished attribute in hLDA is we could have a hierarchial topic tree, with higher branches having more general topics and lower branches having more specialized topics.<br/><br/> ![hlda](/Image/hlda.png)
