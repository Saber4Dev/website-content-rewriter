# -*- coding: utf-8 -*-
"""
ai-model.py

A professional command-line tool to inspect available AI models, their capabilities,
and associated rate limits for different providers (Google Gemini, OpenAI).

This script provides a clear, formatted output of model details, helping developers
understand the constraints and features of each model.

Usage:
    - For Gemini:
      python ai-model.py --provider gemini [--export result.txt]

    - For OpenAI:
      python ai-model.py --provider openai [--export result.txt]

The script will first try to use environment variables (GEMINI_API_KEY, OPENAI_API_KEY).
If an environment variable is not found or is empty, it will securely prompt for the API key.
"""

import os
import argparse
import sys
import getpass
from datetime import datetime
from typing import List, Optional

# --- Provider-Specific Rate Limit Information ---
# This data is based on public documentation and may change.
# It is included here to provide a quick reference.
# Last Updated: 2026-01-11

GEMINI_RATE_LIMITS = {
    "gemini-1.5-pro-latest": {"requests_per_minute": "5 (Free Tier)"},
    "gemini-1.5-flash-latest": {"requests_per_minute": "15 (Free Tier)"},
    "gemini-1.0-pro": {"requests_per_minute": "60 (Standard)"},
    "default": {"requests_per_minute": "See official documentation for the latest details."}
}

OPENAI_RATE_LIMITS = {
    "gpt-4o-mini": {"requests_per_minute": "10,000 (Tier 1)", "tokens_per_minute": "600,000"},
    "gpt-4o": {"requests_per_minute": "5,000 (Tier 1)", "tokens_per_minute": "300,000"},
    "gpt-4-turbo": {"requests_per_minute": "5,000 (Tier 1)", "tokens_per_minute": "300,000"},
    "gpt-3.5-turbo": {"requests_per_minute": "10,000 (Tier 1)", "tokens_per_minute": "1,000,000"},
    "default": {"requests_per_minute": "Varies by model and tier. See official documentation."}
}


def get_api_key(provider_name: str) -> str:
    """
    Gets the API key from environment variables or prompts the user securely.
    Exits the script if no key is provided.
    """
    env_var = f"{provider_name.upper()}_API_KEY"
    api_key = os.environ.get(env_var)
    
    if api_key and api_key.strip():
        print(f"✅ Found API key in environment variable '{env_var}'.")
        return api_key
    
    print(f"ⓘ Environment variable '{env_var}' not found or is empty.")
    try:
        api_key = getpass.getpass(f"🔑 Please enter your {provider_name.capitalize()} API key: ")
    except (getpass.GetPassWarning, EOFError):
        print("\n⚠️  Warning: Could not hide API key input. Falling back to standard input.")
        api_key = input(f"🔑 Please enter your {provider_name.capitalize()} API key: ")
    
    if not api_key or not api_key.strip():
        print(f"\n❌ Error: No {provider_name.capitalize()} API key provided. Exiting.")
        sys.exit(1)
        
    return api_key


def get_header(provider_name: str) -> List[str]:
    """Generates a professional header for the tool's output."""
    return [
        "=" * 80,
        f"🔬 AI Model & Rate Limit Inspector: {provider_name.upper()}",
        f"🗓️  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
    ]


def get_footer() -> List[str]:
    """Generates a footer with a disclaimer about rate limits."""
    return [
        "\n" + "-" * 80,
        "⚠️  Disclaimer: Rate limits are based on public documentation and may not reflect",
        "   your current usage tier or recent changes. Always consult the official",
        "   provider documentation for the most accurate and up-to-date information.",
        "=" * 80,
    ]


def get_gemini_models_report(api_key: str) -> List[str]:
    """
    Connects to Google Gemini and returns a formatted report of models,
    their details, and known rate limits.
    """
    output = ["\nFetching model information..."]
    try:
        import google.generativeai as genai
    except ImportError:
        return [
            "❌ Error: 'google-generativeai' package not found.",
            "Please install it using: pip install google-generativeai"
        ]

    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        output.append("✅ Successfully connected to Google Gemini.\n")
    except Exception as e:
        return [
            f"❌ Error connecting to Google Gemini: {e}",
            "   Please ensure your API key is valid and has the necessary permissions."
        ]

    output.append(f"{ 'Model Name':<30} | {'Input Tokens':<15} | {'Output Tokens':<15} | {'RPM (Requests/Min)':<20}")
    output.append("-" * 100)

    # Filter and sort models
    generative_models = sorted(
        [m for m in models if 'generateContent' in m.supported_generation_methods],
        key=lambda m: m.name
    )

    for model in generative_models:
        try:
            m = genai.get_model(model.name)
            input_limit = m.input_token_limit
            output_limit = m.output_token_limit
        except Exception:
            input_limit = "N/A"
            output_limit = "N/A"

        rate_limit_info = GEMINI_RATE_LIMITS["default"]["requests_per_minute"]
        for key, limits in GEMINI_RATE_LIMITS.items():
            if key in model.name:
                rate_limit_info = limits["requests_per_minute"]
                break

        output.append(f"{m.name:<30} | {input_limit:<15} | {output_limit:<15} | {rate_limit_info:<20}")
    
    return output


def get_openai_models_report(api_key: str) -> List[str]:
    """
    Connects to OpenAI and returns a formatted report of models, their details,
    and known rate limits.
    """
    output = ["\nFetching model information..."]
    try:
        import openai
    except ImportError:
        return [
            "❌ Error: 'openai' package not found.",
            "Please install it using: pip install openai"
        ]

    try:
        client = openai.OpenAI(api_key=api_key)
        models_response = client.models.list()
        models = models_response.data
        output.append("✅ Successfully connected to OpenAI.\n")
    except Exception as e:
        return [
            f"❌ Error connecting to OpenAI: {e}",
            "   Please ensure your API key is valid and has the necessary permissions."
        ]

    output.append(f"{ 'Model ID':<30} | {'Context Window (Tokens)':<25} | {'RPM (Requests/Min)':<20} | {'TPM (Tokens/Min)':<15}")
    output.append("-" * 105)

    context_windows = {
        "gpt-4o-mini": 128000,
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
    }

    filtered_models = sorted(
        [m for m in models if "gpt" in m.id and "instruct" not in m.id],
        key=lambda m: m.id
    )
    
    for model in filtered_models:
        context_window = "N/A"
        for key, size in context_windows.items():
             if key in model.id:
                context_window = size
                break
        
        rpm_limit = OPENAI_RATE_LIMITS["default"]["requests_per_minute"]
        tpm_limit = "N/A"
        for key, limits in OPENAI_RATE_LIMITS.items():
            if key in model.id:
                rpm_limit = limits.get("requests_per_minute", "N/A")
                tpm_limit = limits.get("tokens_per_minute", "N/A")
                break

        output.append(f"{model.id:<30} | {context_window:<25} | {rpm_limit:<20} | {tpm_limit:<15}")

    return output


def main():
    """Main function to parse arguments, run checks, and handle output."""
    parser = argparse.ArgumentParser(
        description="A tool to inspect AI models and rate limits.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["gemini", "openai"],
        help="The AI provider to inspect."
    )
    parser.add_argument(
        "--export",
        dest="export_file",
        default=None,
        help="Optional: Path to a .txt file to save the output."
    )

    args = parser.parse_args()
    provider = args.provider.lower()
    export_file = args.export_file
    
    # Get the API key securely
    api_key = get_api_key(provider)
    
    output_lines = []

    # Generate the report content
    output_lines.extend(get_header(provider))
    if provider == "gemini":
        output_lines.extend(get_gemini_models_report(api_key))
    elif provider == "openai":
        output_lines.extend(get_openai_models_report(api_key))
    output_lines.extend(get_footer())
    
    # Generate the final output string
    final_output = "\n".join(output_lines)
    
    # Print to console
    print(final_output)
    
    # Export to file if requested
    if export_file:
        try:
            with open(export_file, "w", encoding="utf-8") as f:
                # Write the output without the initial interactive messages
                report_only_output = "\n".join(output_lines[1:])
                f.write(report_only_output)
            print(f"\n✅ Report successfully saved to: {export_file}")
        except IOError as e:
            print(f"\n❌ Error: Failed to write to file {export_file}.")
            print(f"   Reason: {e}")


if __name__ == "__main__":
    main()
