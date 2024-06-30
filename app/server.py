from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from mistralai.models.chat_completion import ChatMessage
import os
import json
api_keys = json.load(open("./api_keys.json"))
os.environ["MISTRAL_API_KEY"] = api_keys["MISTRAL_API_KEY"]

from mistralai.client import MistralClient
from pydantic import BaseModel
from tabs_MIDI.read_tabs_app import Tabs
from tabs_MIDI.midi_generator import Track
import requests
import os
import time

app = FastAPI()

templates = Jinja2Templates(directory="./app/templates")
app.mount("/static", StaticFiles(directory="./app/static"), name="static")


api_key = os.environ.get("MISTRAL_API_KEY")
print(api_key)
client = MistralClient(api_key=api_key)


SYSTEM_PROMPT = """
You are a skilled guitar instructor and music theory expert. Generate the next four beats of a guitar tab based on the user's input.

1. Review the first four beats and key provided by the user.
2. Generate the next four beats that fit the specified key.
"""

USER_PROMPT = """
Key: {key}

Tab: {tab}
"""
def str_table_to_tab(table:str):
    # Split the input into lines
    lines = table.strip().split('\n')
    
    
    lines = [l.split("|")[1:-1] for l in lines]
    # Initialize an array to hold the beats
    beats = []
    for i in range(len(lines[0])):
        beat=[]
        for j in range(6):
            beat.append(lines[j][i])
        beats.append(beat)
    
    return beats
    
def tab_to_str(t, add_string_keys=True):
    tab = ""
    keys = ['e', 'B', 'G' , 'D' ,'A' , 'E']
    for i in range(6):
        if add_string_keys:
            tab+=keys[i] + "|"
        for j in range(len(t)):
            tab += t[j][i] + "|"
        tab += "\n"
    return tab

def correct_table(table:str):
    table = str_table_to_tab(table)
    for i in range(len(table)):
        max_len = max([len(x) for x in table[i]])
        for j in range(len(table[i])):
            table[i][j] = table[i][j].rjust(max_len, "-")
    table = tab_to_str(table)
    return table

def guitarstral_inference(tab:str, key:str, temperature:float=0.7, top_p:float=1):
    user_messages = USER_PROMPT.format(key=key, tab=tab)
    chat_response = client.chat(
        model='ft:open-mistral-7b:7e80780e:20240629:b5caa7c6',
        messages=[ChatMessage(role='system', content=SYSTEM_PROMPT), ChatMessage(role='user', content=user_messages)],
        temperature=temperature,
        top_p=top_p
    )
    return correct_table(chat_response.choices[0].message.content)

def generare_table(table:str, key:str, nb_beats:int, temperature:float, top_p:float):
    tab_table = str_table_to_tab(table)
    for _ in range(0, nb_beats, 4):
        new_table = guitarstral_inference(table, key, temperature, top_p)
        new_tab_table = str_table_to_tab(new_table)
        tab_table += new_tab_table
        table = new_table
    return tab_table


def convert_to_midi(tablature:str, tempo):
    t = Tabs(tablature.split("\n"))
    t.preprocess()
    t.displayTabs()
    t.convertNotes()

    f_name = f"tab_{time.time()}.mid"
    f_local_path = f"./app/static/midis/{f_name}"
    outputTrack = Track(int(tempo))
    outputTrack.midiGenerator(t.a, path=f_local_path)
    
    command = f"timidity {f_local_path}"
    os.system(command)
    return f"http://localhost:8000/static/midis/{f_name}"
    

class ChatMessageRequest(BaseModel):
    message: str
    nb_beats: int
    temperature: float
    top_p: float
    key: str
    tempo: int
    

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(chat_message: ChatMessageRequest):
    message = chat_message.message
    nb_beats = chat_message.nb_beats
    temperature = chat_message.temperature
    top_p = chat_message.top_p
    tempo = chat_message.tempo
    key = chat_message.key
    
    tablature = generare_table(message, key, nb_beats, temperature, top_p)
    tablature = tab_to_str(tablature)
    midi_url = convert_to_midi(tablature, tempo)
    response = {
        "tablature": tablature.replace("\n", "<br>"),
        "midi_url": midi_url
    }
    return response



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
