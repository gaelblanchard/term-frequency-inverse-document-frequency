#Lloyd Massiah
#Gael Blanchard
#Big Data Project 2
#This program computes the term term relevance using a query term to calculate the relevance
#of all other terms in descending order. Only terms starting with "gene_" and "disease_"
#were considered for the term-term relevancy. Output is written to a file
#This project assumes that project2_data.txt is already in the correct directory and
#it requires the query term as a command line argument when running the program

from __future__ import division
import math
from operator import add
from pyspark import SparkContext
from pyspark import SparkConf
import sys, getopt, pprint

if len(sys.argv) == 3:
	query_term = sys.argv[2]
	data_file_path = sys.argv[1]


	appName = "termTerm"
	master = "local"
	conf = SparkConf().setAppName(appName).setMaster(master)
	sc = SparkContext(conf=conf)

	
	def Mapper(x):
		list_ = []
		#filters out all terms not starting with and ending with a certain string to create a dataset of only those elements
		for each in x[1:]:
			if ((each.startswith("gene_") and each.endswith("_gene")) or (each.startswith("disease_") and each.endswith("_disease"))): 
				list_.append((each, x[0]))
		return list_

	def SecondMapper(x):
		docid_and_word = x[0]
		count = x[1]
		docid = docid_and_word[1]
		return (docid, list((docid_and_word[0], count)))

	def CreateTuple(x):
		list_ = []
		tuple_list = x[1]
		for i in range(0, len(tuple_list), 2):
			list_.append((tuple_list[i], tuple_list[i+1]))
		return (x[0], list_)	

	def WordCountPerDoc(x):
		list_ = []
		docid = x[0]
		list_of_tuples = x[1]
		number_of_terms_in_doc = 0
		#find the total number of terms in a document and outputs it as part of the new mapping
		for each_tuple in list_of_tuples:
			number_of_terms_in_doc += each_tuple[1]
		for each in list_of_tuples:
			list_.append(((each[0], docid), (each[1], number_of_terms_in_doc)))
		return list_

	def ThirdMapper(x):
		word_and_doc = x[0]
		word_count_and_total_word_in_doc = x[1]
		#word_and_doc[0] = term_name, word_and_doc[1] = docid, word_count_total...[0] = wordCount, word_count_total..[1] = wordPerDoc
		return (word_and_doc[0], (word_and_doc[1], word_count_and_total_word_in_doc[0], word_count_and_total_word_in_doc[1]))

	def CreateSecondTuple(x):
		list_ = []
		tuple_list = x[1]
		for i in range(0, len(tuple_list), 3):
			list_.append((tuple_list[i], tuple_list[i+1], tuple_list[i+2]))
		return (x[0], list_)

	def CountDocsPerWord(x): 
		list_ = []
		docsPerWord = 0
		tuple_list = x[1]
		for each in tuple_list:
			docsPerWord += 1
		for each in tuple_list:
			#x[0] = term_name, each[0] = docid, each[1] = wordCount, each[2] = wordsPerDoc
			list_.append(((x[0], each[0]), (each[1], each[2], docsPerWord)))
		return list_

	def TfIdf(x):
		term_name_ = x[0][0]
		second_tuple = x[1]
		#second_tuple[0] = wordCount, second_tuple[1] = wordsPerDoc, second_tuple[2] = docsPerWord
		tfidf = second_tuple[0] / second_tuple[1] * math.log(8357/second_tuple[2])
		return (term_name_, tfidf)

	def PrintFinalOutput(x):#prints the results to an output file
		list_ = x
		file = open("Lloyd_Massiah_Gael_Blanchard_Output.txt", "w")
		file.write('\n'.join('%s %s' % x for x in list_))

	def GetQueryVector(x, query_term):
		if x[0] == query_term:
			return True
		return False

	def FilterOutQuery(x):
		if x[0] == query_term:
			return False
		else:
			return True

	def SemanticSimilarity(x):
		A_vector = x[0][1]
		B_vector = x[1][1]
		A_denominator = 0
		B_denominator = 0
		A_B_denominator = 0
		A_B_numerator = 0
		semantic_similarity = 0

		#calculates the denominator part for the A vector 
		for i in range(0, len(A_vector), 1):
			A_denominator += A_vector[i] * A_vector[i]

		A_denominator = math.sqrt(A_denominator)	

		#calculates the denominator part for the B vector
		for i in range(0, len(B_vector), 1):
			B_denominator += B_vector[i] * B_vector[i]

		B_denominator = math.sqrt(B_denominator)	

		#makes the vectors equal sized in order to allow multiplication of both vectors
		if len(B_vector) <= len(A_vector):
			difference = len(A_vector) - len(B_vector)
			for i in xrange(difference):
				B_vector.append(0)
		elif len(A_vector) <= len(B_vector):
			difference = len(B_vector) - len(A_vector)
			for i in xrange(difference):
				A_vector.append(0)

		#multiplies each element of A and B to find the numerator of the semantic similarity formula
		for i in xrange(len(A_vector)):
			A_B_numerator += A_vector[i] * B_vector[i]

		#calculates the denominator of the semantic similarity formula
		A_B_denominator = A_denominator * B_denominator

		#output is ((A-term, B-term), semantic similarity)
		return (x[1][0], x[0][0]), A_B_numerator/A_B_denominator


	project2_data = sc.textFile(data_file_path)
	key_value = project2_data.map(lambda x: x.split())#splits each term into a list of every string in it

	#flatMap calls the function mapper than outputs a list of tuples of the form (docid, term) and separates them into individual terms
	#map creates a tuple out of every (key, value) pair and makes the pair the key and gives it the value 1 - > ((docid, term), 1)
	#reduceByKey combines the pairs and leaves only unique term per document
	word_count = key_value.flatMap(lambda x: Mapper(x))\
		.map(lambda x: (x, 1))\
		.reduceByKey(lambda x, y: x + y)

	#map calls a second mapper function that manipulates the tuple to emit the form (docid, (word, wordcount))
	#reduceByKey brings all terms together by their key(their respective documents) in the form (docid, (word1, wordcount1), (word2, wordcount2)...)
	#reduceByKey destroys the tuple structure so it is necessary to remap the previous mapping and term each value into a tuple of the form above
	#flatMap calls word counter per doc and determines how many terms are inside each document and emits a pair in the form ((word, docid), (wordcount, wordsPerDoc))
	doc_count = word_count.map(lambda x: SecondMapper(x))\
		.cache()\
		.reduceByKey(lambda x, y: x + y)\
		.map(lambda x: CreateTuple(x))\
		.flatMap(lambda x: WordCountPerDoc(x))

	#map calls ThirdMapper and manipulates the term and changes the key-value pair into the form (word, (docid, wordcount, wordsPerDoc))	
	#afterwards reduceByKey is used and brings all values together based upon the key which is the "term" or "word"
	#create_second_tuple is called because reduceByKey undoes the structure of the tuples, and tuples allow easy manipulation of the data
	#flatMap is used to separate each key-value pair and calculate docsPerWord
	#final output is ((word, docid), (wordcount, wordsPerDoc, docsPerWord))
	word_per_doc = doc_count.map(lambda x: ThirdMapper(x))\
		.cache()\
		.reduceByKey(lambda x, y: x + y) \
		.map(lambda x: CreateSecondTuple(x))\
		.flatMap(lambda x: CountDocsPerWord(x))

	#calculates the tfidf for each term
	#creates the term vector by grouping together each tfidf value using their common key
	#output will be (term, [tfidf1, tfidf2,..tfidfN])	
	tfidf = word_per_doc.map(lambda x: TfIdf(x)).groupByKey()\
		.cache()\
		.map(lambda x: (x[0], list(x[1])))	

	#returns the term vector that corresponds to the query term
	query_vector = tfidf.filter(lambda x: GetQueryVector(x, query_term))

	#filters out the query term vector from the dataset
	#cartesian creates a pair using the query vector and each term vector of the dataset creating the output (query vector, other term vector)
	#for each term vector
	#then map calls SemanticSimilarity which uses the paired term vectors to calculate their semantic similarity
	#in order to sort by value, the key and values had to be switched to call the transformation, sortByKey
	#final output becomes ((query term, term), semantic similarity)
	semantic_similarity_ = tfidf.filter(lambda x: FilterOutQuery(x))\
		.cache()\
		.cartesian(query_vector)\
		.map(lambda x: SemanticSimilarity(x))\
		.map(lambda x: (x[1], x[0]))\
		.sortByKey(False)\
		.map(lambda x: (x[1], x[0]))\

	final_output = semantic_similarity_.collect()
	PrintFinalOutput(final_output)

else: 
	print "Please type the query term."
	