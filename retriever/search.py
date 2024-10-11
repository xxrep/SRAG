import requests
import json


class Retriever:
    def __init__(self, port=8001, url="localhost"):
        self.url = f"http://{url}:{port}"

    def send_request(self, payload, headers=None):
        if headers is None:
            headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                error_message = f"Error occurred with status code: {response.status_code}, response text: {response.text}"
                print(error_message)
                return response.text
        except Exception as e:
            print(f"An error occurred: {e}")
            return str(e)

    def retrieve(self, questions):
        request_config = {
            'questions': questions
        }
        response = self.send_request(request_config)
        return response


# if __name__ == '__main__':
#     retriever = Retriever()
#     questions = ["Laleli Mosque: We need to find out which neighborhood the Laleli Mosque is located in.",
#                  "Esma Sultan Mansion: We need to find out which neighborhood the Esma Sultan Mansion is located in.",
#                  "\"Look At Us Now\": We need to find out which American DJs are associated with this song."]
#     retriever.retrieve(questions)
