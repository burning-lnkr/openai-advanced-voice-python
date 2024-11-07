import os
import json
import time
import base64
import logging
import threading
import queue
from typing import Optional, Callable, Dict, Any

import pyaudio
from websocket import create_connection, WebSocketConnectionClosedException
from dotenv import load_dotenv
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

REENGAGE_DELAY_MS: int = 500


# Load configuration from config.yaml
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: Dictionary containing configuration parameters.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}.")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing the configuration file: {e}")
            raise


config = load_config()

# Configuration Constants (Moved to config.yaml)
DEBUG_LOG_FILE: str = config.get("debug_log_file", "incoming_messages.log")
MODEL_NAME: str = config.get("model_name", "gpt-4o-realtime-preview-2024-10-01")
ASSISTANT_VOICE: str = config.get("assistant_voice", "alloy")
TEMPERATURE: float = config.get("temperature", 0.8)
ASSISTANT_TALKS_FIRST: bool = config.get("assistant_talks_first", False)

# WebSocket URL Template (Model name is now dynamic based on config)
WS_URL_TEMPLATE: str = f"wss://api.openai.com/v1/realtime?model={MODEL_NAME}"

# Output folder configuration
OUTPUT_FOLDER: str = config.get("output_folder", "./saved_files")

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Tools Definition
TOOLS: list = [
    {
        "name": "write_to_console",
        "description": "Write a message to the console.",
        "type": "function",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to write to the console.",
                },
            },
            "required": ["message"],
        },
    },
    {
        "name": "save_to_file",
        "description": "Save user-specified content to a file.",
        "type": "function",
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "The name of the file to save (without extension).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "The extension of the file (e.g., txt, md).",
                },
                "file_content": {
                    "type": "string",
                    "description": "The content to write into the file.",
                },
            },
            "required": ["file_name", "file_extension", "file_content"],
        },
    },
    # Additional tools can be added here
]


class WebSocketClient:
    """
    Handles WebSocket connections, sending, and receiving messages.
    """

    def __init__(
        self,
        api_key: str,
        ws_url: str,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Initializes the WebSocket client.

        :param api_key: API key for authentication.
        :param ws_url: WebSocket URL to connect to.
        :param on_message: Callback function to handle incoming messages.
        """
        self.api_key = api_key
        self.ws_url = ws_url
        self.ws = None
        self.on_message = on_message
        self._stop_event = threading.Event()
        self._recv_thread = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """
        Establishes the WebSocket connection and starts the receiver thread.
        """
        try:
            self.ws = create_connection(
                self.ws_url,
                header=[
                    f"Authorization: Bearer {self.api_key}",
                    "OpenAI-Beta: realtime=v1",
                ],
            )
            logger.info("Connected to WebSocket.")
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise

        # Start the message receiving thread
        self._recv_thread = threading.Thread(target=self._receive_messages, daemon=True)
        self._recv_thread.start()

    def _receive_messages(self) -> None:
        """
        Continuously listens for incoming WebSocket messages and handles them using the callback.
        """
        while not self._stop_event.is_set():
            try:
                if self.ws:
                    message = self.ws.recv()
                    if message and self.on_message:
                        data = json.loads(message)
                        self.on_message(data)
            except WebSocketConnectionClosedException:
                logger.error("WebSocket connection closed.")
                break
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break
        logger.info("Exiting WebSocket receiving thread.")

    def send(self, data: Dict[str, Any]) -> None:
        """
        Sends a JSON-serialized message over the WebSocket.

        :param data: The data dictionary to send.
        """
        try:
            with self._lock:
                if self.ws:
                    message = json.dumps(data)
                    self.ws.send(message)
                    logger.debug(f"Sent message: {message}")
        except WebSocketConnectionClosedException:
            logger.error("Cannot send message. WebSocket connection is closed.")
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def close(self) -> None:
        """
        Closes the WebSocket connection and stops the receiver thread.
        """
        self._stop_event.set()
        if self.ws:
            try:
                self.ws.close()
                logger.info("WebSocket connection closed.")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join()


class AudioHandler:
    """
    Manages audio input/output streams and buffering.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        rate: int = 24000,
        format: int = pyaudio.paInt16,
        allow_interruptions: bool = False,  # New parameter
    ) -> None:
        """
        Initializes the AudioHandler.

        :param chunk_size: Number of audio frames per buffer.
        :param rate: Sampling rate.
        :param format: Audio format.
        :param allow_interruptions: If True, microphone remains active while assistant is speaking.
        """
        self.chunk_size = chunk_size
        self.rate = rate
        self.format = format
        self.allow_interruptions = allow_interruptions  # Store the flag

        self.pyaudio_instance = pyaudio.PyAudio()
        self.mic_queue: queue.Queue = queue.Queue()
        self.audio_buffer = bytearray()

        self.mic_on_at: float = 0.0
        self.mic_active: bool = False

        self._stop_event = threading.Event()
        self._threads: list = []

        # New attributes for handling playback interruption
        self.playback_interrupted = False
        self._playback_lock = threading.Lock()

    def interrupt_playback(self) -> None:
        """
        Interrupts the current audio playback by clearing the buffer and setting a flag.
        """
        with self._playback_lock:
            self.playback_interrupted = True
            self.audio_buffer.clear()
        logger.debug("Playback interrupted due to user speech.")

    def reset_playback(self) -> None:
        """
        Resets the playback interruption flag to allow future audio playback.
        """
        with self._playback_lock:
            if self.playback_interrupted:
                self.playback_interrupted = False
                logger.debug("Playback reset. Ready for new audio.")

    def _mic_callback(
        self, in_data: bytes, frame_count: int, time_info: dict, status: int
    ) -> tuple:
        """
        Callback function for microphone input stream.

        :param in_data: Audio data captured from the mic.
        :param frame_count: Number of frames.
        :param time_info: Time information.
        :param status: Status flags.
        :return: Tuple containing audio data and stream continuation flag.
        """
        current_time = time.time()
        if self.allow_interruptions or current_time > self.mic_on_at:
            if not self.mic_active:
                logger.info("🎙️🟢 Mic active")
                self.mic_active = True
            self.mic_queue.put(in_data)
        else:
            if self.mic_active:
                logger.info("🎙️🔴 Mic suppressed")
                self.mic_active = False
        return (None, pyaudio.paContinue)

    def _spkr_callback(
        self, in_data: bytes, frame_count: int, time_info: dict, status: int
    ) -> tuple:
        """
        Callback function for speaker output stream.

        :param in_data: Not used.
        :param frame_count: Number of frames.
        :param time_info: Time information.
        :param status: Status flags.
        :return: Tuple containing audio data to play and stream continuation flag.
        """
        bytes_needed = frame_count * 2  # Assuming 16-bit audio
        with self._playback_lock:
            if self.playback_interrupted:
                # Send silence to effectively mute the speaker
                audio_chunk = b"\x00" * bytes_needed
                return (audio_chunk, pyaudio.paContinue)

        current_buffer_size = len(self.audio_buffer)

        if current_buffer_size >= bytes_needed:
            audio_chunk = bytes(self.audio_buffer[:bytes_needed])
            self.audio_buffer = self.audio_buffer[bytes_needed:]
            self.mic_on_at = time.time() + REENGAGE_DELAY_MS / 1000
        else:
            audio_chunk = bytes(self.audio_buffer) + b"\x00" * (
                bytes_needed - current_buffer_size
            )
            self.audio_buffer.clear()

        return (audio_chunk, pyaudio.paContinue)

    def _send_mic_audio(self, socket_client: WebSocketClient) -> None:
        """
        Continuously reads from the mic queue and sends audio data over WebSocket.

        :param socket_client: Instance of WebSocketClient to send data through.
        """
        while not self._stop_event.is_set():
            try:
                mic_chunk = self.mic_queue.get(timeout=0.1)
                logger.debug(f"🎤 Sending {len(mic_chunk)} bytes of audio data.")
                encoded_chunk = base64.b64encode(mic_chunk).decode("utf-8")
                payload = {"type": "input_audio_buffer.append", "audio": encoded_chunk}
                socket_client.send(payload)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending mic audio: {e}")

    def receive_audio(self, audio_chunk: bytes) -> None:
        """
        Receives audio data from the assistant and buffers it for playback.

        :param audio_chunk: Audio data received from the assistant.
        """
        self.audio_buffer.extend(audio_chunk)

    def start_streams(self, socket_client: WebSocketClient) -> None:
        """
        Initializes and starts microphone and speaker streams.

        :param socket_client: Instance of WebSocketClient to send audio data through.
        """
        self.mic_stream = self.pyaudio_instance.open(
            format=self.format,
            channels=1,
            rate=self.rate,
            input=True,
            stream_callback=self._mic_callback,
            frames_per_buffer=self.chunk_size,
        )
        self.spkr_stream = self.pyaudio_instance.open(
            format=self.format,
            channels=1,
            rate=self.rate,
            output=True,
            stream_callback=self._spkr_callback,
            frames_per_buffer=self.chunk_size,
        )

        self.mic_stream.start_stream()
        self.spkr_stream.start_stream()

        # Start audio sending thread
        send_thread = threading.Thread(
            target=self._send_mic_audio, args=(socket_client,), daemon=True
        )
        self._threads.append(send_thread)
        send_thread.start()

        logger.info("Audio streams started.")

    def stop_streams(self) -> None:
        """
        Stops all audio streams and associated threads.
        """
        self._stop_event.set()

        if hasattr(self, "mic_stream") and self.mic_stream.is_active():
            self.mic_stream.stop_stream()
            self.mic_stream.close()

        if hasattr(self, "spkr_stream") and self.spkr_stream.is_active():
            self.spkr_stream.stop_stream()
            self.spkr_stream.close()

        self.pyaudio_instance.terminate()

        for thread in self._threads:
            if thread.is_alive():
                thread.join()

        logger.info("Audio streams stopped.")


class RealtimeAssistant:
    """
    Coordinates WebSocket communication and audio handling for the real-time assistant.
    """

    def __init__(
        self,
        api_key: str,
        ws_url: str,
        enable_debug: bool = False,
        allow_interruptions: bool = False,
        assistant_voice: str = "alloy",
        temperature: float = 0.8,
    ) -> None:
        """
        Initializes the RealtimeAssistant.

        :param api_key: API key for OpenAI services.
        :param ws_url: WebSocket URL for real-time communication.
        :param enable_debug: Flag to enable debug logging of all incoming messages.
        :param allow_interruptions: If True, allows user to interrupt the assistant by keeping the mic active.
        :param assistant_voice: The voice to be used by the assistant.
        :param temperature: The temperature setting for the assistant's responses.
        """
        self.api_key = api_key
        self.ws_url = ws_url
        self.enable_debug = enable_debug
        self.allow_interruptions = allow_interruptions  # Store the flag
        self.assistant_voice = assistant_voice
        self.temperature = temperature

        self.assistant_response_text: str = ""

        # Initialize WebSocket client with message handler
        self.ws_client = WebSocketClient(
            api_key=self.api_key, ws_url=self.ws_url, on_message=self.handle_message
        )

        # Initialize AudioHandler with the new flag
        self.audio_handler = AudioHandler(allow_interruptions=self.allow_interruptions)

        # Initialize debug logger if debug is enabled
        if self.enable_debug:
            self._setup_debug_logger()

        # Lock for thread-safe operations
        self._lock = threading.Lock()

    def _setup_debug_logger(self) -> None:
        """
        Sets up a separate logger for debug logging of all incoming messages.
        """
        self.debug_logger = logging.getLogger("DEBUG_LOGGER")
        self.debug_logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # File handler for debug logs
        file_handler = logging.FileHandler(DEBUG_LOG_FILE, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.debug_logger.addHandler(file_handler)

        logger.info(
            f"Debug logging enabled. Incoming messages will be saved to {DEBUG_LOG_FILE}."
        )

    def start(self) -> None:
        """
        Starts the assistant by connecting WebSocket, sending session updates, and starting audio streams.
        """
        # Connect to WebSocket
        self.ws_client.connect()

        # Register tools and configure session
        self._initialize_session()

        # Conditionally start the conversation based on configuration
        if ASSISTANT_TALKS_FIRST:
            message_payload = {"type": "response.create"}
            self.ws_client.send(message_payload)
            logger.info("Assistant initiated the conversation.")

        # Start audio streams
        self.audio_handler.start_streams(self.ws_client)

    def _initialize_session(self) -> None:
        """
        Sends session configuration and initializes the conversation.
        """
        session_update_payload = {
            "event_id": "event_initsessionupdate",
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": (
                    "You are a helpful, witty, and friendly AI assistant. "
                    "Your voice and personality should be warm and engaging. "
                    "You are allowed and encouraged to talk to user about your available tools and their functionality."
                ),
                "voice": self.assistant_voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": self.temperature,
                "max_response_output_tokens": "inf",
            },
        }

        self.ws_client.send(session_update_payload)

    def handle_message(self, message: Dict[str, Any]) -> None:
        """
        Handles incoming WebSocket messages based on their type.

        :param message: The incoming message as a dictionary.
        """
        event_type = message.get("type")

        # Log all incoming messages if debug is enabled
        if self.enable_debug and hasattr(self, "debug_logger"):
            if event_type not in [
                "response.audio.delta",
                "response.audio_transcript.delta",
            ]:
                self.debug_logger.debug(json.dumps(message, indent=2))

        logger.debug(f"Received message: {message}")

        # Handle messages based on exact type matches
        if event_type == "error":
            self._handle_error(message)
        elif event_type == "response.audio.delta":
            self._handle_audio_delta(message)
        elif event_type == "response.audio_done":
            self._handle_audio_done(message)
        elif event_type == "response.audio_transcript.delta":
            self._handle_audio_transcript_delta(message)
        elif event_type == "response.audio_transcript.done":
            self._handle_audio_transcript_done(message)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            self._handle_input_audio_transcript_done(message)
        elif event_type == "response.delta":
            self._handle_response_delta(message)
        elif event_type == "response.function_call_arguments.done":
            self._handle_function_call(message)
        elif event_type == "function_call_result":
            self._handle_function_call_result(message)
        elif event_type == "input_audio_buffer.speech_started":
            self._handle_user_speech_started(message)  # New handler
        else:
            logger.debug(f"Ignored unhandled event type: {event_type}")

    def _handle_user_speech_started(self, message: Dict[str, Any]) -> None:
        """
        Handles the event when the user starts speaking.

        :param message: The speech_started event message.
        """
        logger.debug("User has started speaking. Interrupting assistant playback.")
        self.audio_handler.interrupt_playback()

    def _handle_error(self, message: Dict[str, Any]) -> None:
        """
        Handles error messages.

        :param message: The error message dictionary.
        """
        error = message.get("error", "Unknown error")
        logger.error(f"Error from WebSocket: {error}")

    def _handle_audio_delta(self, message: Dict[str, Any]) -> None:
        """
        Handles audio delta messages from the assistant.

        :param message: The audio delta message.
        """
        audio_content = base64.b64decode(message.get("delta", ""))
        self.audio_handler.receive_audio(audio_content)
        self.audio_handler.reset_playback()  # Reset playback flag for new audio

    def _handle_audio_done(self, message: Dict[str, Any]) -> None:
        """
        Handles the event when the assistant finishes sending audio.

        :param message: The audio done message.
        """
        logger.debug("AI finished speaking.")

    def _handle_audio_transcript_delta(self, message: Dict[str, Any]) -> None:
        """
        Handles transcription delta messages from the assistant.

        :param message: The transcription delta message.
        """
        delta = message.get("delta", "")
        if delta:
            self.assistant_response_text += delta

    def _handle_audio_transcript_done(self, message: Dict[str, Any]) -> None:
        """
        Handles the event when the assistant finishes sending a transcription.

        :param message: The transcription done message.
        """
        transcript = message.get("transcript", "")
        logger.info(f"Assistant said: {transcript}")
        self.assistant_response_text = ""

    def _handle_input_audio_transcript_done(self, message: Dict[str, Any]) -> None:
        """
        Handles the event when the assistant finishes sending a transcription.

        :param message: The transcription done message.
        """
        transcript = message.get("transcript", "")
        logger.info(f"User said: {transcript}")
        self.assistant_response_text = ""

    def _handle_response_delta(self, message: Dict[str, Any]) -> None:
        """
        Handles text delta messages from the assistant.

        :param message: The text delta message.
        """
        text_delta = message.get("delta", "")
        logger.info(f"Assistant: {text_delta}")

    def _handle_function_call(self, message: Dict[str, Any]) -> None:
        """
        Handles function call arguments generated by the assistant.

        :param message: The function call arguments message.
        """
        function_name = message.get("name")
        function_arguments = message.get("arguments")
        call_id = message.get("call_id")

        if function_name and function_arguments:
            logger.info(
                f"Function call received: {function_name} with arguments {function_arguments}"
            )
            try:
                function_args = json.loads(function_arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding function arguments: {e}")
                function_args = {}

            self._execute_function(function_name, function_args, call_id)

    def _handle_function_call_result(self, message: Dict[str, Any]) -> None:
        """
        Placeholder for handling function call results. Can be extended as needed.

        :param message: The function call result message.
        """
        logger.debug("Received function call result. Currently not handled.")

    def _execute_function(
        self, function_name: str, function_args: Dict[str, Any], call_id: str
    ) -> None:
        """
        Executes a function based on the function name and arguments.

        :param function_name: The name of the function to execute.
        :param function_args: The arguments for the function.
        :param call_id: The call identifier for tracking.
        """
        if function_name == "write_to_console":
            self._write_to_console(function_args.get("message"), call_id)
        elif function_name == "save_to_file":
            self._save_to_file(
                function_args.get("file_name"),
                function_args.get("file_extension"),
                function_args.get("file_content"),
                call_id,
            )
        else:
            logger.warning(f"Unknown function: {function_name}")

    def _write_to_console(self, message: str, call_id: str) -> None:
        """
        Writes a message to the console and notifies the assistant.

        :param message: The message to write.
        :param call_id: The call identifier for tracking.
        """
        if message:
            logger.info(f"Console Message: {message}")
            response_payload = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "Message written to console successfully.",
                },
            }
            self.ws_client.send(response_payload)

            followup_payload = {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"],
                    "instructions": (
                        "Now call the next tool, or continue the conversation if you are done "
                        "or require more input for the task."
                    ),
                },
            }
            self.ws_client.send(followup_payload)
        else:
            logger.error("No message provided for write_to_console function.")

    def _save_to_file(
        self, file_name: str, file_extension: str, file_content: str, call_id: str
    ) -> None:
        """
        Saves the provided content to a file within the OUTPUT_FOLDER.

        :param file_name: The name of the file to save (without extension).
        :param file_extension: The extension of the file (e.g., txt, md).
        :param file_content: The content to write into the file.
        :param call_id: The call identifier for tracking.
        """
        if not all([file_name, file_extension, file_content]):
            logger.error("Missing parameters for save_to_file function.")
            return

        # Sanitize the file name and extension to prevent path traversal
        safe_file_name = os.path.basename(file_name)
        safe_file_extension = os.path.basename(file_extension)

        # Ensure that the file extension does not contain harmful characters
        if not safe_file_extension.isalnum():
            logger.error("Invalid file extension provided.")
            return

        # Construct the full file path
        full_file_name = f"{safe_file_name}.{safe_file_extension}"
        file_path = os.path.join(OUTPUT_FOLDER, full_file_name)

        # Resolve the absolute path and ensure it's within OUTPUT_FOLDER
        try:
            absolute_output_folder = os.path.abspath(OUTPUT_FOLDER)
            absolute_file_path = os.path.abspath(file_path)

            if not absolute_file_path.startswith(absolute_output_folder):
                logger.error("Attempted path traversal detected.")
                return

            # Write the content to the file
            with open(absolute_file_path, "w", encoding="utf-8") as f:
                f.write(file_content)

            logger.info(f"File saved successfully at: {absolute_file_path}")

            # Notify the assistant about the successful operation
            response_payload = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": f"File '{full_file_name}' saved successfully in '{OUTPUT_FOLDER}'.",
                },
            }
            self.ws_client.send(response_payload)

            followup_payload = {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"],
                    "instructions": (
                        "Now call the next tool, or continue the conversation if you are done "
                        "or require more input for the task."
                    ),
                },
            }
            self.ws_client.send(followup_payload)

        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            # Notify the assistant about the failure
            response_payload = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": f"Failed to save file '{full_file_name}'. Error: {str(e)}",
                },
            }
            self.ws_client.send(response_payload)

    def stop(self) -> None:
        """
        Stops the assistant by closing WebSocket and audio streams.
        """
        logger.info("Shutting down RealtimeAssistant.")

        self.audio_handler.stop_streams()
        self.ws_client.close()


def main() -> None:
    """
    Entry point for the assistant. Initializes and starts the RealtimeAssistant.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set in the environment.")
        return

    ws_url = WS_URL_TEMPLATE  # Dynamic URL based on config

    # Determine if debug mode is enabled via config
    debug_mode = config.get("debug_mode", False)

    # Determine if interruption mode is enabled via config
    allow_interruptions = config.get("allow_interruptions", False)

    # Initialize the RealtimeAssistant with debug logging and interruption mode if desired
    assistant = RealtimeAssistant(
        api_key=api_key,
        ws_url=ws_url,
        enable_debug=debug_mode,
        allow_interruptions=allow_interruptions,  # Pass the new flag
        assistant_voice=ASSISTANT_VOICE,
        temperature=TEMPERATURE,
    )

    try:
        assistant.start()
        logger.info("RealtimeAssistant started. Press Ctrl+C to stop.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupt received. Shutting down...")

    finally:
        assistant.stop()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
