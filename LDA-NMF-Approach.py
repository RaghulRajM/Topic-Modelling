# Importing necessary libraries
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
from gensim.corpora.dictionary import Dictionary
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Reading the dataset
df = pd.read_csv("/kaggle/input/unstructured-l0-nlp-hackathon/data.csv")

# Preprocessing text data
df['clean_text'] = df['text'].str.replace("[^a-zA-Z#]", " ")
stopwords_list = stopwords.words('english')
punctuations = set(string.punctuation)

def clean_text(text):
    text = ' '.join([x.lower() for x in word_tokenize(text) if x.lower() not in stopwords_list and len(x) > 1])
    text = ' '.join([x.lower() for x in word_tokenize(text) if x.lower() not in punctuations and len(x) > 3])
    text = ' '.join([x.lower() for x in word_tokenize(text) if nltk.pos_tag([x])[0][1].startswith("NN") or nltk.pos_tag([x])[0][1].startswith("JJ")])
    return text.strip()

df["clean_text"] = df.text.apply(lambda text: clean_text(str(text)))

# Topic Modeling using LDA
cleaned_text_list = df.clean_text.apply(lambda clean_text: word_tokenize(clean_text))
gensim_dict = Dictionary(cleaned_text_list)
doc_term_matrix = [gensim_dict.doc2bow(text) for text in cleaned_text_list]

lda_model = LdaModel(corpus=doc_term_matrix, num_topics=5, id2word=gensim_dict, passes=10, random_state=1, alpha=0.1, eta=0.4)

# Extracting topics using LDA
def get_lda_topics(model, num_topics):
    word_dict = {'Topic ' + str(i): [x.split('*') for x in words.split('+')] for i, words in model.show_topics(num_topics, 10)}
    return pd.DataFrame.from_dict(word_dict)

lda_topics = get_lda_topics(lda_model, 5)
print(lda_topics)

# Evaluating LDA model
df_doc_top = pd.DataFrame()
final_list = []

for index in range(len(df.clean_text)):
    word_id_dict = dict(lda_model.get_document_topics(doc_term_matrix[index]))
    word_score_list = [word_id_dict.get(index, 0) for index in range(5)]
    final_list.append(word_score_list)

df_doc_top = pd.DataFrame(final_list, columns=['Topic ' + str(i) for i in range(1, 6)])
df_doc_top["Dominant_Topic"] = df_doc_top.idxmax(axis=1).tolist()
df_doc_top["Topic_Probability"] = df_doc_top.max(axis=1).tolist()

# Creating a document-topic dataframe
document_df = df_doc_top.reset_index().rename(columns={"index": "Document"})[["Document", "Dominant_Topic", "Topic_Probability"]]

# Mapping topic names
document_df["Dominant_Topic"] = document_df["Dominant_Topic"].map({"Topic 5": "Automobiles", "Topic 4": "room_rentals",
                                                                     "Topic 3": "glassdoor_reviews", "Topic 2": "sports_news",
                                                                     "Topic 1": "tech_news"})

# Creating initial submission CSV
initial_submission = pd.concat([df['Id'], document_df['Dominant_Topic']], axis=1)
initial_submission = initial_submission.set_index("Id").rename(columns={"Dominant_Topic": "topic"})
initial_submission.to_csv("initial_submission.csv")

# Topic Modeling using NMF
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf_matrix = vectorizer.fit_transform(df['clean_text'])
nmf_model = NMF(n_components=5, random_state=1)

# Fitting NMF model
nmf_model.fit(tf_matrix)

# Extracting features (topics) and transforming the document-term matrix
tfidf_feature_names = vectorizer.get_feature_names_out()
nmf_feature_names = [tfidf_feature_names[i] for i in nmf_model.components_.argsort(axis=1)[:, -1:-6:-1]]

# Evaluating NMF model
df_nmf_topics = pd.DataFrame(nmf_model.transform(tf_matrix), columns=["Topic " + str(i) for i in range(1, 6)])
df_nmf_topics["Dominant_Topic"] = df_nmf_topics.idxmax(axis=1).tolist()
df_nmf_topics["Topic_Probability"] = df_nmf_topics.max(axis=1).tolist()

# Creating a document-topic dataframe for NMF
document_nmf_df = df_nmf_topics.reset_index().rename(columns={"index": "Document"})[["Document", "Dominant_Topic", "Topic_Probability"]]

# Mapping topic names for NMF
document_nmf_df["Dominant_Topic"] = document_nmf_df["Dominant_Topic"].map({0: "Automobiles", 1: "room_rentals",
                                                                             2: "glassdoor_reviews", 3: "sports_news",
                                                                             4: "tech_news"})

# Creating NMF submission CSV
nmf_submission = pd.concat([df['Id'], document_nmf_df['Dominant_Topic']], axis=1)
nmf_submission = nmf_submission.set_index("Id").rename(columns={"Dominant_Topic": "topic"})
nmf_submission.to_csv("nmf_submission.csv")
