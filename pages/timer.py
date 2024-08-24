import streamlit as st
import time
from IPython.display import HTML

def pomodoro_timer(pomodoro_length, short_break_length, long_break_length, long_break_interval):
    
    pomodoro_count = 0
    running = False
    start_button = st.button("Start Pomodoro Timer")

    reset_button = st.button("Reset Timer") # create a reset button

    if start_button:
        running = True
        while running:
            # Pomodoro session
            st.write(f"Pomodoro Session #{pomodoro_count + 1}")
            timer_length = countdown(pomodoro_length, "Time Remaining -")

            # Increment pomodoro count and check if long break is needed
            pomodoro_count += 1
            if pomodoro_count % long_break_interval == 0:
                st.write("Time for a long break!")
                timer_length = countdown(long_break_length, "Long Break Duration - ")
            else:
                st.write("Time for a short break!")
                timer_length = countdown(short_break_length, "Short Break Duration - ")

            # Check if reset button is clicked
            if reset_button:
                running = False
                timer_text.empty()

def countdown(timer_length, title):
    timer_length *= 60
    while timer_length:
        mins, secs = divmod(timer_length, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        timer_text.write(f"<h1>{title}{timeformat}</h1>",unsafe_allow_html=True) # update the text field with the current time and title
        time.sleep(1)
        timer_length -= 1
    st.balloons()

# Streamlit app code
st.title("Pomodoro Timer")

music_file = st.file_uploader("Upload Music File (mp3 format)", type=["mp3"])

if music_file is not None:
    st.audio(music_file, format="audio/mp3", start_time=0)

# st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
st.write("Welcome to the Pomodoro Timer! Please enter the following values:")

pomodoro_length = st.number_input("Pomodoro Length (minutes)", min_value=1, step=1, value=25)
short_break_length = st.number_input("Short Break Length (minutes)", min_value=1, step=1, value=5)
long_break_length = st.number_input("Long Break Length (minutes)", min_value=1, step=1, value=15)
long_break_interval = st.number_input("Long Break Interval (Pomodoros)", min_value=1, step=1, value=4)

timer_text = st.empty() # create an empty text field to display the timer

pomodoro_timer(pomodoro_length, short_break_length, long_break_length, long_break_interval)