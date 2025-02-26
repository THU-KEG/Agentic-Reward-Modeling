from openai import OpenAI
import requests

class APIModel:
    def __init__(self, base_url, model_name, api_key="EMPTY"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
    
    def search(self, query, max_tokens=512):
        return self.generate(query, max_tokens=max_tokens)

    def generate(self, query, max_tokens=1024, temperature=0.0):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response = chat_completion.choices[0].message.content
        except Exception as e:
            print(e)
            response = "NA"
        return response

    def generate_chat(self, messages, max_tokens=1024, temperature=0.0):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            print(e)
            response = "NA"
        return response
    
    def safety_check(self, query):
        try:
            chat_completion = self.client.moderations.create(
                model=self.model_name,
                input=query
            )
            import pdb; pdb.set_trace()
            return dict(chat_completion.results[0].categories), chat_completion.results[0].flagged
        except Exception as e:
            print(e)
            response = {"NA": 1}
        return response


class LocalAPIModel:
    def __init__(self, base_url, model_name, api_key="EMPTY"):
        self.base_url = base_url
        self.model_name = model_name
    
    def search(self, query, max_tokens=512):
        return self.generate(query, max_tokens=max_tokens)

    def generate(self, query, max_tokens=1024, temperature=0.0):
        return self.generate_chat(
            [{"role": "user", "content": query}],
            max_tokens=max_tokens,
            temperature=temperature
        )

    def generate_chat(self, messages, max_tokens=1024, temperature=0.0):
        request_data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.0
        }
        response = requests.post(
            self.base_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=900 
        )
        if response.status_code == 200:
            resp_json = response.json()
            content = resp_json['choices'][0]['message']['content'].strip()
            return content
        else:
            print(
                f"Failed to fetch response: {response.status_code}, {response.text}"
            )
            return None
    
     