# User-Comment-Analysis
In this project, I led a group of 6 student to implement knowledge in NLP to process user comments for Roku Ultra on Amazon and Twitter. We applied methods such as LDA, random forest, sentiment analysis and ANOVA to investigate, and generated data-supported business  insight. This repo shares the data, the presentation, and the code I personally wrote for this project.

## Overview
We are curious in whether operation system updates for ROKU (a prominent player in internet-TV industry in the US)  drive the user to like or dislike the product more. We gathered the data from September of 2020 to Feburary of 2022, which span acorss 2 updates and 3 versions. 

We used LDA topic modelling technique to examine the hidden topic for each version, random forrest to examine the relationship between words and rating, sentiment analysis and ANOVA to detect the trend of comments and word clouds to create visually pleasing graphs.   

### My role in this project...
1. Me and another team member came up with the topic and chose the product (Roku Ultra). We wanted to investigate how user react to a new technological product. I came up with the idea of comparing the updates.
2. I wrote all the codes in this directory and shared them to my team members. 
3. I took time to teach my teammates who are not farmiliar with the ideas like LDA and sentiment analisys. 
4. With another teammate, I help the group to break down the project into achievable steps, distribute works according to individual's strength, and organized meetings.

## Files in this projest:
1. The data tables are in the data folder.
  - 1.1 The amazon_api.ipynb is the file that gathered the amazon reviews using Rainforrest API
  - 1.2 the Data explore.ipynb is where I examined the raw text from Amazon
  - 1.3 the data preparation.ipynb is where I applied cleaning, stop word removal and stemming to the 
      data, getting it ready for modelling
  - 1.4 the tweets_API.py is where I used API to get information on twitter regarding Roku Ultra
  - 1.5 the tweets_clean.py is where I applied claning, stoo word removel and stemming to clean the 
      tweets we gathered.
2. Modelling.
  - 2.1 the modelling.ipynb is where I applied random forrest to predict whether a comment is 
      positive (rating between 4-5) or negative (rating between 1-3) on Amazon.
  - 2.2 the lda_attempt_roku.py is where I applied LDA model to extract hidden topics
  - 2.3 the senti_analysis.py is where I applied Hu&Bing dictionary to calculate the sentiment score
      for each comment, and applied ANOVA to examine whether the difference between different 
      versions is significant
3. Reports.
  - 3.1 the Research Report.pdf is a write up version of our ideas, methods and finding. 
      I personally wrote the LDA and sentiment analysis part, and is responsible for the overall 
      quality and flow of the paper. 
  - 3.2 the ROKU NLP Final Deck.pdf is the deck we used to present in class
  
