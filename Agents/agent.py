import os
from utils.common import situation_question_generation_prompt, generate_storytelling_questions,evaluate_depression_level
from dotenv import load_dotenv
import google.generativeai as gen_ai



load_dotenv()

class Agents:
    def __init__(self):
        self.GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
        gen_ai.configure(api_key=self.GOOGLE_API_KEY)
        self.model = gen_ai.GenerativeModel('gemini-2.0-flash')
        

    
    def situation_question_generation_agent(self,data):
        prompt = situation_question_generation_prompt(data)
        response = self.model.generate_content(prompt)
        print("Response from situation question generation agent:", response.text)
        return response.text
    
    def generate_storytelling_questions_agent(self,data):
        prompt = generate_storytelling_questions(data)
        response = self.model.generate_content(prompt)
        return response.text
    
    def evaluate_user(self,data):
        prompt = evaluate_depression_level(data)
        response = self.model.generate_content(prompt)
        return response.text
    
    

