# OpenAI Advanced Voice Python Assistant

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ⚠️ **Disclaimer:**

**Using the OpenAI Realtime API can incur significant costs. I strongly advise against using personal savings to try this tool, as the expenses may and will outweigh the benefits. Proceed with caution and monitor your usage.**

---

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

**OpenAI Advanced Voice Python Assistant** is a Python-based assistant that leverages OpenAI's Realtime API to facilitate real-time audio interactions. Designed for personal experimentation, this assistant supports live audio conversations, transcriptions of both user and assistant speech, and integrates tool calling capabilities for extended functionality. The project is inspired by and based on the [openai-realtime-py](https://github.com/p-i-/openai-realtime-py) repository. Tested on Win11, aims to be compatible with Linux.

## Features

- **Real-time Voice Interaction:** Live audio conversations with the assistant.
- **Speech Transcription:** Transcription of both user and assistant speech.
- **Tool Calling:** Invoke predefined tools during conversations.
- **Interruptions Support:** Interrupt the assistant while it is speaking.
- **Debug Logging:** Optionally log all incoming messages for debugging purposes.

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

     ```bash
     python -m venv venv
     venv\Scripts\activate
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
     ```

     - `OPENAI_API_KEY`: Your OpenAI API key.

2. **Configuration File**

   - Copy `config.yaml.example` to `config.yaml`:

     ```bash
     cp config.yaml.example config.yaml
     ```

   - Open the `config.yaml` file and adjust the settings as you see fit.

## Usage

1. **Activate Virtual Environment (optional)**

   - **On Windows:**

     ```bash
     venv\Scripts\activate
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

The assistant supports tool calling, allowing it to perform specific functions during the conversation. Available tools include:

- **write_to_console**

  - **Description:** Writes a specified message to the console.
  - **Parameters:** `message` (string) – The message to write.

- **save_to_file**
  - **Description:** Saves user-specified content to a file.
  - **Parameters:**
    - `file_name` (string): Name of the file without extension.
    - `file_extension` (string): File extension (e.g., txt, md).
    - `file_content` (string): Content to write into the file.

_Additional tools can be added by extending the `TOOLS` list in the configuration._

## Logging

If `debug_mode` is enabled in the `config.yaml` file, all incoming messages (excluding certain binary types) will be logged to `incoming_messages.log`. This is useful for debugging and monitoring the assistant's interactions.

## Interruptions

By enabling `allow_interruptions` in the `config.yaml` file, the assistant allows users to interrupt its speech by keeping the microphone active.

## Disclaimer

**OpenAI Advanced Voice Python Assistant** is intended for personal use and experimentation. It is **not** production-ready and may not handle all edge cases correctly. Use at your own risk.

Additionally, using the OpenAI Realtime API can be **costly**. Be aware of rapid credit spending and monitor your usage to avoid unexpected charges.

## License

This project is licensed under the [MIT License](LICENSE).

---
