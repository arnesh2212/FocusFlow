FILEPATH = "/home/arnesh/Desktop/TECH SPECTRE/pages/tdos.txt"
class functions:
    def get_tdos(filepath=FILEPATH):
        with open(filepath, 'r') as file_local:
            tdos_local = file_local.readlines()
        return tdos_local

    def write_tdos(tdos_arg, filepath=FILEPATH):
        with open(filepath, 'w') as file:
            file.writelines(tdos_arg)

import streamlit as st

# Load existing tasks
tdos = functions.get_tdos()

def add_todo():
    todo = st.session_state["new_todo"] + "\n"
    tdos.append(todo)
    functions.write_tdos(tdos)

def mark_todo_complete(index):
    tdos[index] = tdos[index].replace('\n', '') + " [Completed]\n"
    functions.write_tdos(tdos)

def delete_completed_tasks():
    tdos[:] = [todo for todo in tdos if "[Completed]" not in todo]
    functions.write_tdos(tdos)

st.title("Do's")
st.subheader("LEVEL UP your productivity")

# Input for new task
new_todo = st.text_input(label="New Task", placeholder="Enter new task", on_change=add_todo, key="new_todo")

# Display active tasks
st.subheader("Active Tasks")
for index, todo in enumerate(tdos):
    # Check if the task is marked as completed
    completed = "[Completed]" in todo
    
    if not completed:
        # Checkbox to mark the task as complete
        chx = st.checkbox(todo, key=todo)
        if chx:
            mark_todo_complete(index)
            st.experimental_rerun()

# Display completed tasks
st.subheader("Completed Tasks")
for index, todo in enumerate(tdos):
    # Check if the task is marked as completed
    completed = "[Completed]" in todo
    
    if completed:
        # Display completed tasks with strikethrough
        st.markdown(f"<del>{todo}</del>", unsafe_allow_html=True)
# Button to delete all completed tasks
if any("[Completed]" in todo for todo in tdos):
    st.button("Delete All Completed Tasks", on_click=delete_completed_tasks)