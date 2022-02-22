# Data Collection Methodology

I found these datasets through Google's [Dataset Search](https://datasetsearch.research.google.com) tool. 
All of the datasets contain both tweet texts and their corresponding labels, except for **hatespeech_id_label.csv** and **NAACL_SRW_2016.csv**, which only contain tweet IDs along with labels. I created the script **twitter_api_text_grabber.py** as a pipeline that leverages the [Twitter API](https://developer.twitter.com/en/docs/twitter-api) to add a corresponding text column to the two aforementioned datasets. The ammended datasets resulting from this pipeline are named **hatespeech_id_label_text.csv** and **NAACL_SRW_2016_text.csv**. Note that some tweets are no longer publicly available, so the ammended text datasets may have less samples than their ID only counterparts. 

For each dataset, I have included below its source url, filename, and rough number of samples. For more information about a given dataset, please see its source url. 

## - Datasets - 

https://www.kaggle.com/vkrahul/twitter-hate-speech \
**train_E6oV3lV.csv** *(~32k samples)*

https://data.world/thomasrdavidson/hate-speech-and-offensive-language \
**labeled_data.csv** *(~25k samples)*

https://zenodo.org/record/2657374#.YQM6IC1h06V and [Twitter API](https://developer.twitter.com/en/docs/twitter-api)\
**hatespeech_id_label.csv** *(~100k samples)* \
**hatespeech_id_label_text.csv** *(~55k samples)*

https://github.com/ZeerakW/hatespeech and [Twitter API](https://developer.twitter.com/en/docs/twitter-api) \
**NAACL_SRW_2016.csv** *(~17k samples)* \
**NAACL_SRW_2016_text.csv** *(~10k samples)*


