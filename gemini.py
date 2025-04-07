# -*- coding: utf-8 -*-

"""
gemini.py

A command-line tool to interact with the Google Gemini API.

Usage:
  gemini.py "Your prompt text here"
  echo "Your prompt text via pipe" | gemini.py
  gemini.py < file_with_prompt.txt
  gemini.py (will read from stdin interactively)

Requires the 'google-generativeai' library (pip install google-generativeai)
and the GEMINI_API_KEY environment variable to be set.
"""

import google.generativeai as genai
import argparse
import sys
import os
import textwrap

def call_gemini(prompt_text, model_name="gemini-2.0-flash"):
    """
    Sends the prompt text to the Gemini API and returns the response.

    Args:
        prompt_text (str): The prompt to send to the API.
        model_name (str): The name of the Gemini model to use.

    Returns:
        str: The text content of the Gemini API response, or an error message.
    """
    try:
        # Configure the API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY environment variable not set."
        genai.configure(api_key=api_key)

        # Add a system instruction for concise answers
        system_prompt = "Context: Linux. Provide a concise response:\n"
        full_prompt = system_prompt + prompt_text

        # Create the model instance
        model = genai.GenerativeModel(model_name)

        # Send the prompt and get the response
        response = model.generate_content(full_prompt)

        # Extract and return the text part of the response
        if not response.candidates:
            return "Error: No content generated (check safety settings or prompt)."
        return response.text.strip()

    except Exception as e:
        return f"Error calling Gemini API: {e}"

def main():
    """
    Main function to parse arguments, get prompt, call API, and print result.
    """
    parser = argparse.ArgumentParser(
        description="Send a prompt to the Google Gemini API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              gemini.py "What is the capital of France?"
              gemini.py "Translate 'hello world' to Spanish"
              gemini.py "Write a python function to calculate factorial"
              gemini.py "show all files"
              echo "Summarize this text" | gemini.py
              cat document.txt | gemini.py
              gemini.py < prompt.txt
            """)
    )

    # Use nargs='*' to capture all positional arguments as a list
    parser.add_argument(
        'prompt_parts',
        nargs='*',
        help='The prompt text (if not provided, reads from stdin).'
    )
    parser.add_argument(
        '--model',
        default="gemini-1.5-flash-latest",
        help='The Gemini model to use (default: gemini-1.5-flash-latest).'
    )


    args = parser.parse_args()

    prompt = ""
    if args.prompt_parts:
        # Join the list of arguments into a single string prompt
        prompt = " ".join(args.prompt_parts)
    else:
        # Check if stdin is connected to a terminal (interactive input)
        if sys.stdin.isatty():
            print("Enter prompt (press Ctrl+D on Linux/macOS or Ctrl+Z then Enter on Windows to end):", file=sys.stderr)
        # Read all data from stdin until EOF
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("Error: No prompt provided either as argument or via stdin.", file=sys.stderr)
            parser.print_help(sys.stderr)
            sys.exit(1)

    # Call the Gemini API
    result = call_gemini(prompt, args.model)

    # Print the result to stdout
    print(result)

    # Exit with error code if the result indicates an error
    if result.lower().startswith("error:"):
        sys.exit(1)

if __name__ == "__main__":
    main()
