# OpenAI Advanced Voice Python Assistant

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Tools](#tools)
- [Logging](#logging)
- [Interruptions](#interruptions)
- [Disclaimer](#disclaimer)
- [License](#license)

## Overview

**OpenAI Advanced Voice Python Assistant** is a single-file Python script that simulates an advanced voice interaction experience like in OpenAI's apps, using OpenAI's realtime API. Designed for personal use and experimentation, this assistant allows for real-time audio communication, transcription of both user and assistant speech, and supports tool calling with example integrations. Nearly all of it was written by o1-mini.

**Note:** This project is inspired by and based on the [openai-realtime-py](https://github.com/p-i-/openai-realtime-py) repository. Tested on Win11, aims to be compatible with Linux.

## Features

- **Real-time Voice Interaction:** Live audio conversations with the assistant.
- **Speech Transcription:** Transcribes both user and assistant speech in real-time.
- **Tool Calling:** Supports invoking predefined tools during conversations.
- **Interruptions Support:** Optionally allows users to interrupt the assistant while it is speaking.
- **Debug Logging:** Optionally logs all incoming messages for debugging purposes.
- **Example Tool:** Includes an example tool (`write_to_console`) that demonstrates tool integration.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **Virtual Environment (recommended)**

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/openai-advanced-voice-python.git
   cd openai-advanced-voice-python
   ```

2. **Create a Virtual Environment (optional)**

   - **On Windows:**

     ```
     python -m venv venv
     venv\Scripts\activate.ps1
     ```

   - **On Linux:**

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Environment Variables**

   - Copy `.env.example` to `.env`:

     ```bash
     cp .env.example .env
     ```

   - Open the `.env` file and configure the following variables:

     ```
     OPENAI_API_KEY=your-openai-key
     DEBUG_MODE=0
     ALLOW_INTERRUPTION=1
     ```

     - `OPENAI_API_KEY`: Your OpenAI API key.
     - `DEBUG_MODE`: Set to `1` to enable debug logging, `0` to disable.
     - `ALLOW_INTERRUPTION`: Set to `1` to allow microphone interruptions during assistant speech, `0` to disable.

## Usage

1. **Activate Virtual Environment(optional)**

   - **On Windows:**

     ```bash
     venv\Scripts\activate.ps1
     ```

   - **On Linux:**

     ```bash
     source venv/bin/activate
     ```

2. **Run the Assistant**

   ```bash
   python realtime_assistant.py
   ```

3. **Interact**

   - Speak into your microphone to communicate with the assistant.
   - The assistant will respond in real-time with audio and text transcriptions.

## Tools

The assistant supports tool calling, allowing it to perform specific functions during the conversation. An example tool `write_to_console` is included, which writes messages to the console.

### Example Tool: `write_to_console`

- **Description:** Writes a specified message to the console.
- **Function Call Parameters:**
  - `message` (string): The message to write.

**Usage:**

During a conversation, you might ask assistant to invoke this tool to display messages directly in the console.

## Logging

If `DEBUG_MODE` is enabled in the `.env` file, all incoming messages (excluding certain binary types) will be logged to `incoming_messages.log`.

## Interruptions

By enabling `ALLOW_INTERRUPTION` in the `.env` file, the assistant allows users to interrupt its speech by keeping the microphone active.

## Disclaimer

**OpenAI Advanced Voice Python Assistant** is intended for personal use and experimentation. It is **not** production-ready and may not handle all edge cases correctly. Use at your own risk.

Additionally, using the OpenAI realtime API can be **costly**. Be aware of rapid credit spending and monitor your usage to avoid unexpected charges.

## License

This project is licensed under the [MIT License](LICENSE).

---
