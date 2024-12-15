from transformers import pipeline
import whisper
import numpy as np
import time
import matplotlib.pyplot as plt
from openai import OpenAI
import os
import langchain
import pymupdf
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import messages_to_dict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from plots import *
from base64_converter import *
from desc_generator import *
from vidtoaud import *
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(api_key = os.getenv("GROQ_API_KEY"), model = "llama-3.3-70b-versatile", temperature = 0)
# Load the ASR model from Hugging Face
def transcribe_audio(audio_path):
    
  client = OpenAI(api_key = os.getenv("OPEN_AI_API_KEY"))

  audio_file= open(audio_path, "rb")
  transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file,
  response_format="verbose_json",
  timestamp_granularities=["word", "segment"]
  )
  
  confidence_thresholds = {
        "Perfect": 85,
        "Very Good": 70,
        "Good": 60,
        "Fair": 50
    }
  word_clarity_matrix = []
  pronunciation_accuracy_matrix = {
        "Perfect": 0, 
        "Very Good": 0, 
        "Good": 0, 
        "Fair": 0, 
        "Poor": 0
    }
  total_words = 0

    # Process segments if available
  segments = transcription.segments if hasattr(transcription, 'segments') else []
  words = transcription.words if hasattr(transcription, 'words') else []

  return {
        "text": transcription.text,
        "segments": segments,
        "words": words,
        "word_clarity_matrix": word_clarity_matrix,
        "pronunciation_accuracy_matrix": pronunciation_accuracy_matrix,
        "total_words": total_words
    }

# Example usage
audio_path = output_mp3 #"E:/Projects and Internships/plugin/harvard.mp3" #output_mp3
audio_path_wav = output_wav #"E:/Projects and Internships/plugin/harvard.wav" #output_wav

system_message = """
You are an expert at speaking and understanding english.
For the given text, return an asnwer pointing out all grammatical issues in the text
return a json based output, with fields as (Category of the grammatical issue, index range of the error text)
Keep in mind, the categories for grammatical errors can be (Articles error, tenses, prepositions, conjunctions, adjectives and verbs related)
the text is
"*{input}*"

Further calculate the error density for the text.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", "{input}")
    ]
)

model = prompt | llm | StrOutputParser()
transcription_result = transcribe_audio(audio_path)

input = transcription_result["text"]
response = model.invoke({"input":input})

prompt2 = ChatPromptTemplate.from_messages([
    ("system","You are an expert at extracting numbers from a given text. For a given text, just output the number of errors in it. The output should be a single integer only."),
    ("human","{response}")
])
chain2 = prompt2 | llm | StrOutputParser()
errors = int(chain2.invoke({"response":response}))
total_words = len(input.split())

def categorize_confidence(confidence, thresholds):
    if confidence > thresholds["Perfect"]:
        return "Perfect"
    elif confidence > thresholds["Very Good"]:
        return "Very Good"
    elif confidence > thresholds["Good"]:
        return "Good"
    elif confidence > thresholds["Fair"]:
        return "Fair"
    else:
        return "Poor"

confidence_thresholds = {
        "Perfect": 85,
        "Very Good": 70,
        "Good": 60,
        "Fair": 50
    }
      
word_clarity_matrix = transcription_result["word_clarity_matrix"]
pronunciation_accuracy_matrix = transcription_result["pronunciation_accuracy_matrix"]

total_words = transcription_result["total_words"]
segments = transcription_result["segments"]
for segment in segments:
  start = segment.start
  end = segment.end
  raw_confidence = segment.no_speech_prob * 100

  pronunciation_quality = categorize_confidence(raw_confidence, confidence_thresholds)
  word_clarity_matrix.append(raw_confidence)
  pronunciation_accuracy_matrix[pronunciation_quality] += 1
  total_words += 1

pronunciation_accuracy_percentage = {key: (count / total_words) * 100 for key, count in pronunciation_accuracy_matrix.items()}

average_word_clarity = np.mean(word_clarity_matrix) if word_clarity_matrix else 0

fig_punc, ax = plt.subplots(figsize=(10, 6))
ax.bar(pronunciation_accuracy_matrix.keys(), pronunciation_accuracy_matrix.values(),
        color=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"])
ax.set_xlabel("Clarity Categories")
ax.set_ylabel("Word Count")
ax.set_title("Pronunciation Accuracy by Clarity Categories")

text = input

mfccs, fig1 = compute_mfcc(audio_path)
zcr, fig2 = compute_zero_crossing_rate(audio_path)
gunning_index = compute_gunning_index(text)
error_density , fig3= compute_error_density(errors, total_words)
wps, fig4 = compute_words_per_second(audio_path, text)
avg_pause, fig5 = compute_pause_pattern(audio_path)
res, smr = classify_audio(audio_path_wav)

b1 = fig_to_base64(fig1)
b2 = fig_to_base64(fig2)
b3 = fig_to_base64(fig3)
b4 = fig_to_base64(fig4)
b5 = fig_to_base64(fig5)
b6 = fig_to_base64(fig_punc)
b = [b1, b2, b3, b4, b5, b6]

json_data = {
    "mfccs": [np.float32(mfccs.mean()), b1],
    "zcr": [np.float32(zcr.mean()), b2],
    "gunning_index": [np.float32(gunning_index)],
    "error_density": [np.float32(error_density), b3],
    "wps": [np.float32(wps), b4],
    "avg_pause": [np.float32(avg_pause), b5],
    "punc_acc":[b6]
}

if(res):
  json_data["smr"] = [res, np.float32(smr)] 
else:
  json_data["smr"] = ["No SMR Data avaiable", 0]
  
desc = ""
for i in range(0, 6):
  print(i)
  desc = desc + description_generator(b[i])
  
system1 = """
You are an expert at analyzing english speaking skills of people, given the descriptions of 
characteristics of the audios, 
(The characteristics are different plost developed by studying the audio data features)
Using the 
""
{desc}, and the values 
{metric1}-{score1}, {metric2}-{score2}, {metric3}-{score3}, {metric4}-{score4}
{metric5}-{score5}, {metric6}-{score6}, {metric7}-{data_avail}-{score7}
"
(Relevant plots were used to create the description, you have to reference the scores of the same metrics too for this task)
Develop a feedback for the user, explaining her how her english currently is.
Mention pros and cons for the user. Ignore the metric if data is not available.
"""

prompt1 = ChatPromptTemplate.from_messages([
    ("system", system1),
])

chain1 = prompt1 | llm

from langchain_core.runnables import RunnableParallel # or whatever LLM you're using
# Your existing prompt and system message definitions
systempros = """
Extract pros of the {feedback}, output only the list of the pros, ""NOTHING ELSE"
"""
systemcons = """
Extract cons of the {feedback}, output only the list of the cons, ""NOTHING ELSE"
"""
systemsummary = """
Generate a 50 words summary for the {feedback}.
"""

# Create prompt templates - note the change here
promptpros = ChatPromptTemplate.from_template(systempros)
promptcons = ChatPromptTemplate.from_template(systemcons)
promptsummary = ChatPromptTemplate.from_template(systemsummary)

# Create individual chains
chainpros = promptpros | llm | StrOutputParser()
chaincons = promptcons | llm | StrOutputParser()
chainsummary = promptsummary | llm | StrOutputParser()

# Create a parallel runnable to execute chains in parallel
parallel_chains = RunnableParallel(
    pros=chainpros,
    cons=chaincons,
    summary=chainsummary
)

# Simple synchronous function to run the chains
def run_parallel_chains(chain1, input_data):
    # First, get the output from the initial chain
    initial_output = chain1.invoke({"desc":desc, 
                                    "metric1": "mfccs", "score1":json_data["mfccs"][0], 
                                    "metric2": "Zero Crossing Rate", "score2":json_data["zcr"][0],
                                    "metric3": "Words per second", "score3":json_data["wps"][0],
                                    "metric4": "Gunnning Index", "score4":json_data["gunning_index"][0],
                                    "metric5": "Grammatical Error Density", "score5":json_data["error_density"][0],
                                    "metric6": "Average Pause", "score6":json_data["avg_pause"][0],
                                    "metric7":"SMR", "data_avail":json_data["smr"][0], "score7": json_data["smr"][1]})
    
    # Then run the parallel chains with the initial output
    responses = parallel_chains.invoke({"feedback":initial_output})
    
    # Unpack the responses
    response_pros = responses['pros']
    response_cons = responses['cons']
    response_summary = responses['summary']
    
    return response_pros, response_cons, response_summary

input_data = "The audio is great"
response_pros, response_cons, response_summary = run_parallel_chains(chain1, input_data)

system_imp_pan = """

You are an expert at english speaking and understanding.
Given the {desc} of user's english audio, create a comprehensive 
improvement plan for the user, enabling her to improve her english skills.
(Do not mention weaknesses and strengths, straightaway start with the plan, no introduction line in te response)
"""
prompt_imp_pan = ChatPromptTemplate.from_template(system_imp_pan)
chain_imp_pan = prompt_imp_pan | llm | StrOutputParser()
imp_plan = chain_imp_pan.invoke({"desc":desc})

json_data2 = {
  "pros": response_pros,
  "cons": response_cons,
  "summary": response_summary,
  "improvement_plan": imp_plan
}

print(imp_plan)