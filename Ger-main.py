import streamlit as st

import os
import deepl
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

from numpy import dot
from numpy.linalg import norm
import numpy as np

#OPENAI API 키 저장
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
DEEPL_API_KEY = st.secrets["DEEPL_API_KEY"]

# LLM 모델 설정
chatgpt=OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
chatgpt2=ChatOpenAI(model_name="gpt-4o", temperature=0)
google = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
claude = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

deeplt = deepl.Translator(DEEPL_API_KEY)

# 임베딩
embedding = OpenAIEmbeddings()

# cosine similarity
def cos_sim(A, B):
   return dot(A, B)/(norm(A)*norm(B))

# 번역 - 역번역 모듈

def Translate(Eng1):
   #번역 모듈
   Q1="다음 문장을 독일어로 번역해줘,  \
      독일인들이 자주 쓰는 표현을 사용해서 쉽게 이해할 수 있게 하고 \
      되도록 원문과 같은 문장 형태를 가지도록 만들어주고, \
      번역 결과만 보여줘  : " + Eng1

   Google1 = google.invoke(Q1)
   GPT1 = chatgpt.invoke(Q1)
   Claude1 = claude.invoke(Q1)
   #GPT2 = chatgpt2.invoke(Q1)
   Deepl_R = str(deeplt.translate_text(Eng1, target_lang="de"))

   # print(Deepl_R)
   # print(type(Deepl_R))
   
   # 역번역 모듈
   Q2Goo="Please translate the following sentence into English,  \
    using common American expressions for easy understanding \
    and maintaining the original sentence structure as much as possible. \
    Show only the translated sentence  : " + Google1.content

   Q2Gpt1="Please translate the following sentence into English,  \
    using common American expressions for easy understanding \
    and maintaining the original sentence structure as much as possible. \
    Show only the translated sentence  : " + GPT1 

   # Q2Gpt2="다음 문장을 영어로 번역해줘,  \
   #    미국인들이 자주 쓰는 표현을 사용해서 쉽게 이해할 수 있게 하고 \
   #    되도록 원문과 같은 문장 형태를 가지도록 만들어주고, \
   #    번역 결과만 보여줘  : " + GPT2.content
   
   Q2Deepl="Please translate the following sentence into English,  \
    using common American expressions for easy understanding \
    and maintaining the original sentence structure as much as possible. \
    Show only the translated sentence  : " + Deepl_R

   Q2Claud="Please translate the following sentence into English,  \
    using common American expressions for easy understanding \
    and maintaining the original sentence structure as much as possible. \
    Show only the translated sentence  : " + Claude1.content

   BGoogle1 = google.invoke(Q2Goo)
   BGPT1 = google.invoke(Q2Gpt1)
   #BGPT2 = google.invoke(Q2Gpt2)
   BDeepl = google.invoke(Q2Deepl)
   BClaude1 = google.invoke(Q2Claud)

   # 유사도 검증 모듈
   em_Eng1 = embedding.embed_query(Eng1)
   em_goo = embedding.embed_query(BGoogle1.content)
   em_gpt1 = embedding.embed_query(BGPT1.content)
   em_claud = embedding.embed_query(BClaude1.content)
   #em_gpt2 = embedding.embed_query(BGPT2.content)
   em_deepl = embedding.embed_query(BDeepl.content)

   sGoogle1 = cos_sim(em_Eng1, em_goo)
   sGPT1 = cos_sim(em_Eng1, em_gpt1)
   sClaude1 = cos_sim(em_Eng1, em_claud)
   #sGPT2 = cos_sim(em_Eng1, em_gpt2)
   sDeepl = cos_sim(em_Eng1, em_deepl)

   st.header("Forward-Backward Translantion", divider='rainbow') 
   #
   # Column for GPT3.5
   #
   Kor1, Back1, Sim1 = st.columns([4,4,1])

   with Kor1:
      st.markdown("**GPT3.5**")
      st.write(GPT1)

   with Back1:
      st.markdown("**Back-Translation (google)**")
      st.write(BGPT1.content)

   with Sim1:
      st.markdown("**Similar**")
      st.write(round(sGPT1,3))

   # #
   # # Column for GPT4o
   # #
   # Kor2, Back2, Sim2 = st.columns([4,4,1])

   # with Kor2:
   #    st.markdown("**GPT4o**")
   #    st.write(GPT2.content)

   # with Back2:
   #    st.markdown("&nbsp; ")
   #    st.write(BGPT2.content)

   # with Sim2:
   #    st.markdown("&nbsp; ")
   #    st.write(round(sGPT2,3))

   #
   # Column for Deepl
   #
   Kor2, Back2, Sim2 = st.columns([4,4,1])

   with Kor2:
      st.markdown("**DeepL**")
      st.write(Deepl_R)

   with Back2:
      st.markdown("&nbsp; ")
      st.write(BDeepl.content)

   with Sim2:
      st.markdown("&nbsp; ")
      st.write(round(sDeepl,3))

   #
   # Column for Gemini
   #
   Kor3, Back3, Sim3 = st.columns([4,4,1])

   with Kor3:
      st.markdown("**Google - Gemini**")
      st.write(Google1.content)

   with Back3:
      st.markdown("&nbsp; ")
      st.write(BGoogle1.content)

   with Sim3:
      st.markdown("&nbsp; ")
      st.write(round(sGoogle1,3))

   #
   # Column for Claud
   #
   Kor4, Back4, Sim4 = st.columns([4,4,1])

   with Kor4:
      st.markdown("**Claude3**")
      st.write(Claude1.content)

   with Back4:
      st.markdown("&nbsp; ")
      st.write(BClaude1.content)

   with Sim4:
      st.markdown("&nbsp; ")
      st.write(round(sClaude1,3))

   # Your Choice
   
   Kor1 = st.text_area("Input your sentence (German)", max_chars=2000, height=50,value="")
   if Kor1=="":
      st.warning ("원하는 독일어 번역 문장을 넣어주세요") 
   else:
      QKor1="다음 문장을 한국어로 번역해줘,  \
         한국인들이 자주 쓰는 표현을 사용해서 쉽게 이해할 수 있게 하고 \
         되도록 원문과 같은 문장 형태를 가지도록 만들어주고, \
         번역 결과만 보여줘  : " + Kor1

      BKor1 = google.invoke(QKor1)
      BDeepl2 = str(deeplt.translate_text(BKor1.content, target_lang="ko"))
      st.write("한국어 번역 (google) :", BKor1.content)
      st.write("한국어 번역 (deepl) :", BDeepl2)

      # 한국어 역번역  
      QKor2="Please translate the following sentence into English,  \
         using common American expressions for easy understanding \
         and maintaining the original sentence structure as much as possible. \
         Show only the translated sentence  : " + BKor1.content
      
      BKor2 = google.invoke(QKor2)
      BDeepl3 = str(deeplt.translate_text(BDeepl2, target_lang="en-us"))

      em_goo = embedding.embed_query(BKor2.content)
      em_deepl2 = embedding.embed_query(BDeepl3)
      st.markdown("___")

      sKor1 = cos_sim(em_Eng1, em_goo)
      sKor2 = cos_sim(em_Eng1, em_deepl2)

      st.write("역번역 (google): ", BKor2.content)
      st.write("유사도 (google): ", sKor1)
      st.write("역번역 (deepl): ", BKor2.content)
      st.write("유사도 (deepl): ", sKor2)


# Streamlit 시작
st.title("Forward-Backward Translantion (DE)")

with st.form("Form 1"):
    Eng1 = st.text_area("Input original sentence (English)", max_chars=2000, height=50,value="")
    s_state=st.form_submit_button("submit")
    if s_state:
        if (input == ""):
            st.warning ("원하는 영어 번역 문장을 넣어주세요") 
        else:
            Translate(Eng1)

