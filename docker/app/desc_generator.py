import base64
import openai
import os
from dotenv import load_dotenv
load_dotenv()

# Assuming you have already set up your OpenAI API key in the 'opai' variable
client = openai.OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# Encode the image
def description_generator(b):
  base64_image = b  # Using the base64 encoded image from variable b1
  prompt = """
  For the given plot, interpret the values
  and develop a description in not more than 5 lines. Get
  into as much depth as possible. The result is to be understood
  by scientists studying the audio file for which the plot is drawn.
  Be to the point and in your response, use the data to explain the 
  degree of difference between different aspects.
  """
  # Create the request payload
  response = client.chat.completions.create(
      model="gpt-4o-mini",  # OpenAI's vision model
      messages=[
          {
              "role": "user",
              "content": [
                  {"type": "text", "text": prompt},
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{base64_image}"
                      }
                  }
              ]
          }
      ],
      max_tokens=300  # Adjust as needed
  )

  # Extract and print the description
  image_description = response.choices[0].message.content
  return image_description