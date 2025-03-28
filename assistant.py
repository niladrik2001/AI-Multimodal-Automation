import base64
import json
import os
import cv2
import numpy as np
import mss
import openai
from threading import Lock, Thread
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

# Load environment variables
load_dotenv()


# ========== Webcam Stream Class ==========

class WebcamStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()
            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()


# ========== Capture Functions ==========

def capture_screen():
    """Captures the desktop screen."""
    with mss.mss() as sct:
        screen = sct.grab(sct.monitors[1])
        img = np.array(screen)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def combine_webcam_and_screen():
    """Combines screen and webcam feed vertically."""
    screen_img = capture_screen()
    webcam_img = webcam_stream.read()

    screen_img = cv2.resize(screen_img, (webcam_img.shape[1], screen_img.shape[0]))
    combined = np.vstack((screen_img, webcam_img))
    return combined


# ========== Assistant Class ==========

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        """Generates and speaks response based on prompt + image."""
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        """Text to speech output using OpenAI API."""
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        """Builds LangChain with prompt and chat memory."""
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions. Your job is to answer 
        questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. 

        Be friendly and helpful. Show some personality.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()
        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


# ========== Initialization ==========

# Start persistent webcam
webcam_stream = WebcamStream().start()

# Load OpenAI model
model = ChatOpenAI(model="gpt-4o")
assistant = Assistant(model)

# Setup Whisper and Mic
recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)


def audio_callback(recognizer, audio):
    """Processes voice input and triggers assistant."""
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        combined_feed = combine_webcam_and_screen()
        encoded_image = cv2.imencode(".jpeg", combined_feed)[1].tobytes()
        base64_image = base64.b64encode(encoded_image)
        assistant.answer(prompt, base64_image)

    except UnknownValueError:
        print("There was an error processing the audio.")


# Start background voice recognition
stop_listening = recognizer.listen_in_background(microphone, audio_callback)

# GUI Loop
while True:
    combined_feed = combine_webcam_and_screen()
    cv2.imshow("Combined Feed", combined_feed)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
webcam_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
