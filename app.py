
import streamlit as st
import streamlit.components.v1 as stc
from newspaper import Article
import nltk
import numpy as np
import pandas as pd
import base64
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization import summarize # TextRank Algorithm
from rouge import Rouge # Evaluate Summary
import spacy
from spacy import displacy
import neattext as nt
import neattext.functions as nfx
from collections import Counter
from textblob import TextBlob
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib

import docx2txt
import pdfplumber
from PyPDF2 import PdfFileReader

matplotlib.use('Agg')
nlp = spacy.load('en')
timestr = time.strftime("%Y%m%d-%H%M%S")

# Function to get summary text analysis
def text_analysis(my_text):
	docx = nlp(my_text)
	alldf = [(token.text, token.shape_, token.pos_, token.tag_, token.lemma_, token.is_alpha, token.is_stop) for token in docx]
	df = pd.DataFrame(alldf, columns = ['Token', 'Shape', 'PoS', 'Tag', 'Lemma', 'Is_Alpha', 'Is_StopWord'])
	return df

# Function to get entity
def get_entities(my_text):
	docx = nlp(my_text)
	ent_df = [(entity.text, entity.label_) for entity in docx.ents]
	return ent_df

# Function to get entities with nice format
HTML_WRAPPER = HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
def render_entities(rawtext):
	docx = nlp(rawtext)
	html = displacy.render(docx, style = 'ent')
	html = html.replace('\n\n', '\n')
	result = HTML_WRAPPER.format(html)
	return result

# Function to evaulate text summary
def evaluate_summary(summary, reference):
	r = Rouge()
	eval_score = r.get_scores(summary, reference)
	eval_score_df = pd.DataFrame(eval_score[0])
	return eval_score_df

# Function to get most common words
def most_word(mytext, num=10):
	word_token = Counter(mytext.split())
	most_df = dict(word_token.most_common(num))
	return most_df

# Function to get Sentiment
def get_sentiment(mytext):
	blob = TextBlob(mytext)
	sentiment = blob.sentiment
	return sentiment

# Function to generate wordcloud
def plot_wordcloud(mytext):
	mywordcloud = WordCloud().generate(mytext)
	fig = plt.figure()
	plt.imshow(mywordcloud, interpolation='bilinear')
	plt.axis('off')
	st.pyplot(fig)

# Function to download results
def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "nlp_result_{}_.csv".format(timestr)
    st.markdown("### ** 📩 ⬇️ Download CSV file **")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click here!</a>'
    st.markdown(href, unsafe_allow_html=True)

# Function to read pdf file
def read_pdf(file):
	pdfReader = PdfFileReader(file)
	count = pdfReader.numPages
	all_page = ""
	for i in range(count):
		page = pdfReader.getPage(i)
		all_page += page.extractText()

	return all_page


# Function to sort the most similar sentences
def index_sort(list_var):
	length = len(list_var)
	list_index = list(range(0, length))

	x = list_var

	for i in range(length):
		for j in range(length):
			if x[list_index[i]] > x[list_index[j]]:
				#swap
				temp = list_index[i]
				list_index[i] = list_index[j]
				list_index[j] = temp

	return list_index

# Create Bots Response
def bot_response(user_input, sentence_list):
	
	user_input = user_input.lower()
	sentence_list.append(user_input)

	bot_response = ''
	cm = CountVectorizer().fit_transform(sentence_list)

	similarity_scores = cosine_similarity(cm[-1], cm)
	similarity_scores_list = similarity_scores.flatten()

	index = index_sort(similarity_scores_list)
	index = index[1:]

	response_flag = 0

	j = 0
	for i in range(len(index)):
		if similarity_scores_list[index[i]] > 0.0 and len(sentence_list[index[0]].split()) > 4:
			bot_response = bot_response + ' ' + sentence_list[index[i]]
			response_flag = 1
			j = j+1

		if j > 3:
			break

	if response_flag == 0:
		bot_response = bot_response + ' ' + "I aplogize. I don't understand."

	sentence_list.remove(user_input)

	return bot_response

def main():

	# st.title("Data Science Application Dashboard")

	menu = ['Home', 'NLP Summarization App', 'NLP Text Analysis', 'NLP Q&A App']
	choice = st.sidebar.selectbox("Menu", menu)

	# Home Page
	if choice == 'Home':
		
		st.markdown("""
						<div align='center'><h1 style="color:blue"><font size="5"> Welcome to Data Science Application Dashboard </font> </h1></div> <br>\
						<p> The purpose of this dashboard is to demonstrate few applications on natural language processing (NLP). NLP is a branch of artifical intelligence \
						that deals with the interaction between computers and humans using language. The goals for NLP is to read, understand, translate, and make sense of \
						the humana langages in a manner that is valuable. </p>

						<p> Three applications are demonstrated in the dashboard. Brief descriptions are shown in below: </p>

						<ol>
							<li> NLP Summarization App - This app will allow you to either enter an article from URL, copy and paste texts, or upload a file (.pdf, .docx, .txt). The app will apply \
							TextRank Algorithm to summarize the article and also evaluate the summary using Rouge scores. </li>
							<li>NLP Text Analysis App - This app will allow you to either enter an article from URL, copy and paste texts, or upload a file (.pdf, .docx, .txt). The app will apply \
							variety of python packages (e.g., spacy, neattext, wordcloud, etc) to analyze the article and generate word statistics, top keywords, sentiment, wordcloud, and more. </li>
							<li> NLP Q&A App - This app will allow you to either enter an article from URL or copy and paste texts. The app will read the article and you can ask any questions \
							related to the article. The app will apply cosine similarity to response few sentences that are closed related to your question. </li>
						</ol> 


						 """, unsafe_allow_html=True)



	# NLP Q&A Page
	elif choice == 'NLP Q&A App':

		st.subheader("Natural Language Processing (NLP) Q&A Application")
		file_format = ['URL','Plain Text']
		files = st.sidebar.selectbox("Input Format", file_format)

		if files == 'URL':

			text_input = st.text_area("Please Enter an URL Article")
			if text_input:

				try:

					#print(text_input)

					# Get Article
					article = Article(text_input.strip())
					article.download()
					article.parse()
					article.nlp()
					corpus  = article.text

					#print(corpus)
					#print(text)

					# Tokenization
					text = corpus
					sentence_list = nltk.sent_tokenize(text)

					# print(text)

					question = st.text_area("Please Enter a question related to the Artcile")

					if question:

						response = bot_response(user_input = question, sentence_list = sentence_list)
						st.subheader(response)

				except:

					st.warning("Please Enter a correct URL with Article")

		elif files == 'Plain Text':

			text_input = st.text_area("Please Enter Text Here")
			if text_input:

				try:

					question = st.text_area("Please Enter a question related to the Artcile")

					sentence_list = nltk.sent_tokenize(text_input)

					# print(sentence_list)

					if question:

						response = bot_response(user_input = question, sentence_list = sentence_list)
						st.subheader(response)

				except:

					st.warning("Please Enter some text")

	# NLP Summarization page
	elif choice == 'NLP Summarization App':

		st.subheader("Natural Language Processing (NLP) Summarization Application")
		file_format = ['URL','Plain Text', 'Upload a File']
		files = st.sidebar.selectbox("Input Format", file_format)

		if files == 'URL':

			text_input = st.text_area("Please Enter an URL Article")
			if st.button("Summarize"):

				try:
					# Get Article
					article = Article(text_input.strip())
					article.download()
					article.parse()
					article.nlp()
					corpus  = article.text

					#print(corpus)
					#print(text)

					# Tokenization
					text = corpus
					sentence_list = nltk.sent_tokenize(text)

					art_text = " ".join(sentence_list)

					with st.beta_expander("Original Text"):
						st.write(art_text)

					#c1, c2 = st.beta_columns(2)

					#with c1:
					#	with st.beta_expander("LexRank Summary"):
					#		pass

					with st.beta_expander("TextRank Summary"):
						textrank_sum = summarize(art_text)
						doc_length = {"Article Word Count": len(art_text), "Summary Word Count": len(textrank_sum)}
						st.write(doc_length)
						st.write(textrank_sum)


						st.info("Rouge Score")
						score = evaluate_summary(textrank_sum, art_text)
						st.dataframe(score)

				except:
					st.warning("Please Enter a correct URL with Article")

		elif files == 'Plain Text':

			text_input = st.text_area("Please Enter Text Here")
			if st.button("Summarize"):

				try:

					sentence_list = nltk.sent_tokenize(text_input)
					art_text = " ".join(sentence_list)

					with st.beta_expander("Original Text"):
						st.write(art_text)

					with st.beta_expander("TextRank Summary"):
						textrank_sum = summarize(art_text)
						doc_length = {"Article Word Count": len(art_text.split()), "Summary Word Count": len(textrank_sum.split())}
						st.write(doc_length)
						st.write(textrank_sum)


						st.info("Rouge Score")
						score = evaluate_summary(textrank_sum, art_text)
						st.dataframe(score)

				except:
					st.warning("Please Enter more sentences")

		elif files == 'Upload a File':

			text_file = st.file_uploader("Please Upload a File", type = ['pdf', 'docx', 'txt'])

			if text_file is not None:
				if text_file.type == 'application/pdf':
					text_input = read_pdf(text_file)

				elif text_file.type == 'text/plain':
					text_input = str(text_input, read(), 'utf-8')

				else:
					text_input = docx2txt.process(text_file)

				try:

					with st.beta_expander("Original Text"):
						st.write(text_input)

					with st.beta_expander("TextRank Summary"):
						textrank_sum = summarize(text_input)
						doc_length = {"Article Word Count": len(text_input.split()), "Summary Word Count": len(textrank_sum.split())}
						st.write(doc_length)
						st.write(textrank_sum)


						st.info("Rouge Score")
						score = evaluate_summary(textrank_sum, text_input)
						st.dataframe(score)

				except:
					st.warning("Please Enter more sentences")

	# NLP Text Analysis
	elif choice == 'NLP Text Analysis':

		st.subheader("Natural Language Processing (NLP) Text Analysis Application")
		file_format = ['URL','Plain Text', 'Upload a File']
		files = st.sidebar.selectbox("Input Format", file_format)

		if files == 'URL':

			text_input = st.text_area("Please Enter an URL Article")
			num_of_tokens = st.sidebar.number_input("Most Common Word", 5, 15)
			if st.button("Analyze"):

				try:
					# Get Article
					article = Article(text_input.strip())
					article.download()
					article.parse()
					article.nlp()
					corpus  = article.text

					# Tokenization
					text = corpus
					sentence_list = nltk.sent_tokenize(text)

					art_text = " ".join(sentence_list)

					# Original Text
					with st.beta_expander("Original Text"):
						st.write(art_text)

					# Text Analysis
					with st.beta_expander("Text Analysis"):
						token_df = text_analysis(art_text)
						st.dataframe(token_df)

					# Entities
					with st.beta_expander("Entities"):
						#ent_check = get_entities(art_text)
						#st.write(ent_check)

						entity_df = render_entities(art_text)
						stc.html(entity_df, height=500, scrolling=True)

					c1, c2 = st.beta_columns(2)

					
					with c1:
						# Word Statistics
						with st.beta_expander("Word Statistics"):
							st.info("Word Statistics")
							docx = nt.TextFrame(art_text)
							st.write(docx.word_stats())

						# Plot Part of Speech
						with st.beta_expander("Plot Part of Speech"):
							fig = plt.figure()
							sns.countplot(token_df['PoS'])
							plt.xticks(rotation=45)
							st.pyplot(fig)

						# Get Sentiment
						with st.beta_expander("Sentiment"):
							sent_result = get_sentiment(art_text)
							st.write(sent_result)
					
					with c2:
						# Most Common Word
						with st.beta_expander("Top Keywords"):
							st.info("Top Keywords/Tokens")
							lower_text = art_text.lower()
							remove_sw = nfx.remove_stopwords(lower_text)
							keyword = most_word(remove_sw, num_of_tokens)
							st.write(keyword)

						# Plot Word Freq
						with st.beta_expander("Plot Top Word Frequency"):
							fig = plt.figure()
							top_word = most_word(remove_sw, num_of_tokens)
							plt.bar(top_word.keys(), top_word.values())
							plt.xticks(rotation=45)
							st.pyplot(fig)

						# Generate WordCloud
						with st.beta_expander("Plot WordCloud"):
							lower_text = art_text.lower()
							remove_sw = nfx.remove_stopwords(lower_text)
							plot_wordcloud(remove_sw)

					#with st.beta_expander("Download Text Analysis Results"):
					#	make_downloadable(token_df)

				except:
					st.warning("Please Enter a correct URL with Article")

		elif files == 'Plain Text':

			text_input = st.text_area("Please Enter Text Here")
			num_of_tokens = st.sidebar.number_input("Most Common Word", 5, 15)
			if st.button("Analyze"):

				try:
					
					sentence_list = nltk.sent_tokenize(text_input)
					art_text = " ".join(sentence_list)

					# Original Text
					with st.beta_expander("Original Text"):
						st.write(art_text)

					# Text Analysis
					with st.beta_expander("Text Analysis"):
						token_df = text_analysis(art_text)
						st.dataframe(token_df)

					# Entities
					with st.beta_expander("Entities"):
						#ent_check = get_entities(art_text)
						#st.write(ent_check)

						entity_df = render_entities(art_text)
						stc.html(entity_df, height=500, scrolling=True)

					c1, c2 = st.beta_columns(2)

					
					with c1:
						# Word Statistics
						with st.beta_expander("Word Statistics"):
							st.info("Word Statistics")
							docx = nt.TextFrame(art_text)
							st.write(docx.word_stats())

						# Plot Part of Speech
						with st.beta_expander("Plot Part of Speech"):
							fig = plt.figure()
							sns.countplot(token_df['PoS'])
							plt.xticks(rotation=45)
							st.pyplot(fig)

						# Get Sentiment
						with st.beta_expander("Sentiment"):
							sent_result = get_sentiment(art_text)
							st.write(sent_result)
					
					with c2:
						# Most Common Word
						with st.beta_expander("Top Keywords"):
							st.info("Top Keywords/Tokens")
							lower_text = art_text.lower()
							remove_sw = nfx.remove_stopwords(lower_text)
							keyword = most_word(remove_sw, num_of_tokens)
							st.write(keyword)

						# Plot Word Freq
						with st.beta_expander("Plot Top Word Frequency"):
							fig = plt.figure()
							top_word = most_word(remove_sw, num_of_tokens)
							plt.bar(top_word.keys(), top_word.values())
							plt.xticks(rotation=45)
							st.pyplot(fig)

						# Generate WordCloud
						with st.beta_expander("Plot WordCloud"):
							lower_text = art_text.lower()
							remove_sw = nfx.remove_stopwords(lower_text)
							plot_wordcloud(remove_sw)

					#with st.beta_expander("Download Text Analysis Results"):
					#	make_downloadable(token_df)

				except:
					st.warning("Please Enter more sentences")

		elif files == 'Upload a File':
			
			text_file = st.file_uploader("Please Upload a File", type = ['pdf', 'docx', 'txt'])
			num_of_tokens = st.sidebar.number_input("Most Common Word", 5, 15)
			if text_file is not None:
				if text_file.type == 'application/pdf':
					text_input = read_pdf(text_file)

				elif text_file.type == 'text/plain':
					text_input = str(text_input, read(), 'utf-8')

				else:
					text_input = docx2txt.process(text_file)

				try:

					# Original Text
					with st.beta_expander("Original Text"):
						st.write(text_input)

					# Text Analysis
					with st.beta_expander("Text Analysis"):
						token_df = text_analysis(text_input)
						st.dataframe(token_df)

					# Entities
					with st.beta_expander("Entities"):
						#ent_check = get_entities(art_text)
						#st.write(ent_check)

						entity_df = render_entities(text_input)
						stc.html(entity_df, height=500, scrolling=True)

					c1, c2 = st.beta_columns(2)

					
					with c1:
						# Word Statistics
						with st.beta_expander("Word Statistics"):
							st.info("Word Statistics")
							docx = nt.TextFrame(text_input)
							st.write(docx.word_stats())

						# Plot Part of Speech
						with st.beta_expander("Plot Part of Speech"):
							fig = plt.figure()
							sns.countplot(token_df['PoS'])
							plt.xticks(rotation=45)
							st.pyplot(fig)

						# Get Sentiment
						with st.beta_expander("Sentiment"):
							sent_result = get_sentiment(text_input)
							st.write(sent_result)
					
					with c2:
						# Most Common Word
						with st.beta_expander("Top Keywords"):
							st.info("Top Keywords/Tokens")
							lower_text = text_input.lower()
							remove_sw = nfx.remove_stopwords(lower_text)
							keyword = most_word(remove_sw, num_of_tokens)
							st.write(keyword)

						# Plot Word Freq
						with st.beta_expander("Plot Top Word Frequency"):
							fig = plt.figure()
							top_word = most_word(remove_sw, num_of_tokens)
							plt.bar(top_word.keys(), top_word.values())
							plt.xticks(rotation=45)
							st.pyplot(fig)

						# Generate WordCloud
						with st.beta_expander("Plot WordCloud"):
							lower_text = text_input.lower()
							remove_sw = nfx.remove_stopwords(lower_text)
							plot_wordcloud(remove_sw)

				except:
					st.warning("The uploaded file does not have enough text")


if __name__ == '__main__':
	main()

