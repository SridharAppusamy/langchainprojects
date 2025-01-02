import requests
import json,os
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-7b-hf"


token_access = os.getenv("HF_TOKEN")
headers = {
    "Authorization": f"Bearer {token_access}",
    'Content-Type':'application/json'
    }

history=[]

def generate_response(prompt):
    history.append(prompt)
    final_prompt="\n".join(history)
    data={
        "inputs":final_prompt
    }

    response=requests.post(API_URL,headers=headers,data=json.dumps(data))

    if response.status_code==200:
        response=response.text
        data=json.loads(response)
        actual_response=data[0]
        return actual_response['generated_text']
    else:
        print("error:",response.text)
 
interface=gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4,placeholder="Enter your Prompt"),
    outputs="text"
)
interface.launch(share=False)
