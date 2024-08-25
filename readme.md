# FOCUS FLOW

## Topic : Personalized Education for <span style="color: #e03e2d;">Neurodiverse Students</span>

Create an Al-driven adaptive learning platform that tailors educational content to the unique needs of neurodiverse students, improving their academic outcomes and overall well-being.

* * *

## Motivation

Education is a fundamental right and a powerful tool for personal and professional growth. However, traditional educational systems often fall short in addressing the diverse needs of neurodiverse students—those with conditions such as autism spectrum disorder (ASD), attention deficit hyperactivity disorder (ADHD), dyslexia, and other cognitive variations. These students may face unique challenges in the classroom, such as difficulties with focus, sensory sensitivities, or alternative learning styles, which are not always accommodated by conventional teaching methods.

The motivation behind developing an AI-driven adaptive learning platform is to bridge this gap and provide a more inclusive, personalized educational experience. By leveraging artificial intelligence, we aim to create a platform that tailors educational content and teaching strategies to each student's individual needs. This approach not only enhances engagement and understanding but also promotes a sense of achievement and well-being among neurodiverse learners.

*Key Motivations:*

1.  *Personalized Learning:* Neurodiverse students often require customized learning approaches that cater to their unique strengths and challenges. Our platform uses AI to adapt educational content in real-time, ensuring that each student receives instruction that suits their learning style and pace.
    
2.  *Improved Academic Outcomes:* By addressing the specific needs of neurodiverse students, the platform aims to improve academic performance and retention. Personalized feedback and targeted support help students overcome barriers and achieve their full potential.
    
3.  *Enhanced Well-Being:* Education is not only about academic success but also about emotional and psychological well-being. The platform is designed to create a positive and supportive learning environment, reducing stress and fostering a sense of accomplishment.
    
4.  *Inclusivity:* Our goal is to make education accessible to all students, regardless of their cognitive differences. By integrating adaptive technologies, we aim to promote an inclusive educational landscape where every student can thrive.
    

By focusing on these core motivations, we believe our AI-driven adaptive learning platform will make a meaningful difference in the lives of neurodiverse students, paving the way for a more equitable and supportive educational experience.


## Proposed Solution
1. Attention Monitoring System
Real-Time Feedback: Tracks attention using computer vision and machine learning, providing personalized, non-intrusive alerts.
Adaptive Alerts: Offers tailored prompts to help ADHD students stay focused, with the option to adjust the study environment as needed.
2. RAG-Based LLM Chatbot for Study Assistance
Interactive Learning: Breaks down complex topics into digestible parts, using real-time data retrieval to offer accurate and relevant information.
Emotional Support: Recognizes signs of stress and provides motivational tips or relaxation suggestions to enhance the learning experience.
3. LLM-Based Social Interaction Chatbot
Simulated Social Scenarios: Helps students practice social skills by interacting with a chatbot that mimics peer behavior.
Group Study Facilitation: Organizes and mediates group study sessions, promoting collaboration and peer learning.
4. Enhanced Pomodoro Timer
Custom Soundscapes: Allows students to create personalized audio tracks for a more relaxing and focused study session.
Attention Integration: Syncs with the attention monitoring system to dynamically adjust break times and focus sessions.
5. Advanced TO-DO List
AI Task Recommendations: Suggests task priorities based on the student's deadlines and study habits.
Cross-Platform Sync: Ensures that the TO-DO list is accessible across all devices.
Gamification: Introduces a reward system to encourage task completion and boost motivation.



## Architecture
1. Attention Monitoring System
Mediapipe & OpenCV: Implemented real-time attention tracking using Mediapipe for facial recognition and OpenCV for processing the video feed. These tools work together to monitor students' focus during study sessions.
PyTorch with Mediapipe: Developed and trained models in PyTorch to analyze where the user is looking. The system uses this data to determine if the student is distracted and provides real-time alerts to help them refocus.
2. RAG-Based LLM Chatbot for Study Assistance
RAG (Retrieval-Augmented Generation): Integrated RAG to enhance the chatbot’s ability to provide accurate, contextually relevant information by retrieving data from external sources during interactions.
Gemini: Used Gemini to improve the chatbot’s natural language understanding, enabling it to better interpret and respond to complex queries.
Chroma DB: Employed Chroma DB as the vector database for efficient embedding management and fast retrieval of relevant information.
3. LLM-Based Social Interaction Chatbot
LLM with Gemini and RAG: Combined LLM with Gemini and RAG to create a chatbot that simulates realistic peer interactions, helping students practice social skills in a controlled environment.


## Installation Procedure

- install streamlit
- pip install
    - opencv-python
    - numpy
    - mediapipe
    - collections
    - csv
    - socket
    - argparse
    - pygame
    - joblib
    - google
    - google.generativeai
    - dotenv
    - langchain
    - langchain_google_genai
    - Ipython
    - PyTorch

### Run this to load forntend and backend for viewing

bash
streamlit run home.py


## Developers :

[Arnesh Batra](https://github.com/arnesh2212) , [Anushk kumar](https://github.com/berserk-23115) , [Govind kumar](https://github.com/darkyll) , [Krish Marwah](https://github.com/krishmarwah)

NOTE : If error comes in NeuroPal, just reload the site and it will work fine