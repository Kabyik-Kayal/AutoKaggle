#!/usr/bin/env python3
"""
AutoKaggle Model Selection Demo

This script demonstrates how to use different model configurations
with the latest Anthropic Claude models and OpenAI models.
"""

import json
import os
import sys

def load_config():
    """Load current config"""
    with open('multi_agents/config.json', 'r') as f:
        return json.load(f)

def save_config(config):
    """Save config"""
    with open('multi_agents/config.json', 'w') as f:
        json.dump(config, f, indent=2)

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text):
    """Print formatted section"""
    print(f"\n{text}")
    print("-" * 70)

def show_current_config():
    """Show current model configuration"""
    config = load_config()
    mapping = config.get('agent_model_mapping', {})
    
    print_section("Current Agent Model Configuration")
    for agent, model in mapping.items():
        print(f"  {agent:<20} → {model}")

def apply_profile(profile_name, profile_config):
    """Apply a configuration profile"""
    config = load_config()
    config['agent_model_mapping'] = profile_config
    save_config(config)
    print(f"✓ Applied profile: {profile_name}")
    show_current_config()

def profile_premium():
    """Profile 1: Maximum Capability"""
    profile = {
        "Reader": "gpt-4o-mini",
        "Planner": "claude-sonnet-4-6",
        "Developer": "gpt-4o",
        "Reviewer": "gpt-4o-mini",
        "Summarizer": "gpt-4o-mini"
    }
    apply_profile("Premium (Maximum Capability)", profile)

def profile_balanced():
    """Profile 2: Balanced (Recommended)"""
    profile = {
        "Reader": "claude-haiku-4-5-20251001",
        "Planner": "claude-sonnet-4-6",
        "Developer": "gpt-4o",
        "Reviewer": "claude-haiku-4-5-20251001",
        "Summarizer": "claude-haiku-4-5-20251001"
    }
    apply_profile("Balanced (Recommended) ⭐", profile)

def profile_budget():
    """Profile 3: Budget"""
    profile = {
        "Reader": "claude-haiku-4-5-20251001",
        "Planner": "claude-sonnet-4-6",
        "Developer": "claude-sonnet-4-6",
        "Reviewer": "claude-haiku-4-5-20251001",
        "Summarizer": "claude-haiku-4-5-20251001"
    }
    apply_profile("Budget (Cost-Optimized)", profile)

def profile_all_claude():
    """Profile 4: All Claude"""
    profile = {
        "Reader": "claude-haiku-4-5-20251001",
        "Planner": "claude-sonnet-4-6",
        "Developer": "claude-sonnet-4-6",
        "Reviewer": "claude-sonnet-4-6",
        "Summarizer": "claude-haiku-4-5-20251001"
    }
    apply_profile("All Claude (Anthropic Only)", profile)

def profile_original():
    """Profile 5: Original OpenAI"""
    profile = {
        "Reader": "gpt-4o-mini",
        "Planner": "gpt-4o",
        "Developer": "gpt-4o",
        "Reviewer": "gpt-4o-mini",
        "Summarizer": "gpt-4o-mini"
    }
    apply_profile("Original (All OpenAI)", profile)

def profile_custom():
    """Allow user to create custom profile"""
    print_section("Create Custom Profile")
    
    agents = ["Reader", "Planner", "Developer", "Reviewer", "Summarizer"]
    models = {
        "1": "gpt-4o",
        "2": "gpt-4o-mini",
        "3": "claude-sonnet-4-6",
        "4": "claude-haiku-4-5-20251001",
        "5": "gpt-4o-mini",
    }
    
    print("Available models:")
    for key, model in models.items():
        print(f"  {key}: {model}")
    
    profile = {}
    for agent in agents:
        print(f"\nSelect model for {agent}:")
        choice = input("  Enter number (1-5): ").strip()
        model = models.get(choice, "gpt-4o")
        profile[agent] = model
    
    apply_profile("Custom Profile", profile)

def show_model_info():
    """Show model information"""
    print_section("Latest Anthropic Models (2025)")
    
    models_info = {
        "claude-sonnet-4-6": {
            "capability": "High ⭐⭐⭐⭐",
            "speed": "Medium ⚡",
            "cost": "Balanced 💵",
            "best_for": "Balanced tasks, general purpose, planning, development"
        },
        "claude-haiku-4-5-20251001": {
            "capability": "High ⭐⭐⭐⭐",
            "speed": "Medium ⚡",
            "cost": "Medium 💵",
            "best_for": "Balanced tasks, general purpose, planning"
        },
        "claude-haiku-4-5-20251001": {
            "capability": "Very Good ⭐⭐⭐",
            "speed": "Very Fast 🚀",
            "cost": "Cheapest 💲",
            "best_for": "Fast responses, simple tasks, reading"
        },
        "gpt-4o": {
            "capability": "Most Capable ⭐⭐⭐⭐⭐",
            "speed": "Fast ⚡⚡",
            "cost": "Expensive 💸",
            "best_for": "Complex reasoning, coding, maximum capability"
        }
    }
    
    for model, info in models_info.items():
        print(f"\n{model}")
        for key, value in info.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")

def show_menu():
    """Show main menu"""
    print_header("AutoKaggle Model Selection Tool")
    print_section("Choose an action")
    
    print("1. View current configuration")
    print("2. Apply Profile 1: Premium (Maximum Capability)")
    print("3. Apply Profile 2: Balanced (Recommended) ⭐")
    print("4. Apply Profile 3: Budget (Cost-Optimized)")
    print("5. Apply Profile 4: All Claude (Anthropic Only)")
    print("6. Apply Profile 5: Original (All OpenAI)")
    print("7. Create Custom Profile")
    print("8. Show Model Information")
    print("9. Show Usage Examples")
    print("0. Exit")
    
    return input("\nEnter your choice (0-9): ").strip()

def show_usage_examples():
    """Show usage examples"""
    print_section("Usage Examples")
    
    examples = [
        ("Default OpenAI", "python framework.py --competition titanic --model gpt-4o"),
        ("Latest Claude Sonnet", "python framework.py --competition titanic --model claude-sonnet-4-6"),
        ("Best Balance: Claude Sonnet", "python framework.py --competition titanic --model claude-sonnet-4-6"),
        ("Fast Claude Haiku", "python framework.py --competition titanic --model claude-haiku-4-5-20251001"),
    ]
    
    for desc, cmd in examples:
        print(f"\n{desc}:")
        print(f"  $ {cmd}")

def main():
    """Main menu loop"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    while True:
        choice = show_menu()
        
        if choice == "0":
            print("\n✓ Exiting. Your configuration has been applied.")
            break
        elif choice == "1":
            show_current_config()
        elif choice == "2":
            profile_premium()
            print("\n✓ Profile applied successfully. Exiting...")
            break
        elif choice == "3":
            profile_balanced()
            print("\n✓ Profile applied successfully. Exiting...")
            break
        elif choice == "4":
            profile_budget()
            print("\n✓ Profile applied successfully. Exiting...")
            break
        elif choice == "5":
            profile_all_claude()
            print("\n✓ Profile applied successfully. Exiting...")
            break
        elif choice == "6":
            profile_original()
            print("\n✓ Profile applied successfully. Exiting...")
            break
        elif choice == "7":
            profile_custom()
            print("\n✓ Profile applied successfully. Exiting...")
            break
        elif choice == "8":
            show_model_info()
        elif choice == "9":
            show_usage_examples()
        else:
            print("✗ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
