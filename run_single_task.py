#!/usr/bin/env python3
"""
Minimal script to run a single data science task for testing in OSWorld.
"""

import os
import sys
import json
import subprocess


def run_single_data_science_task():
    """Run a single data science task for testing."""

    # Update with the task ID you want to test
    task_id = "ds131"
    print(f"\nRunning data science task: {task_id}\n")

    # Check if the OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("The OpenAI API key is not set.")
        print('Please run: export OPENAI_API_KEY="your-api-key-here"')
        return False

    # Create a temporary configuration for the selected task
    test_config = {"data_science": [task_id]}
    temp_config = "temp_single_task.json"
    with open(temp_config, "w") as f:
        json.dump(test_config, f, indent=2)

    # Path to your VMware virtual machine
    vmx_path = r"/home/ivlr/vmware/Ubuntu 64-bit/Ubuntu 64-bit.vmx"

    try:
        # Build the command
        cmd = [
            "python", "run_data_science.py",
            "--model", "gpt-4o",
            "--provider_name", "vmware",
            "--path_to_vm", vmx_path,
            "--test_all_meta_path", temp_config,
            "--max_steps", "10",
            "--temperature", "0.1",
            "--observation_type", "screenshot_a11y_tree"
        ]

        print("Executing the following command:")
        print(" ".join(cmd))
        print()

        # Run the command
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode == 0:
            print(f"The task {task_id} completed successfully.")
            print("You can find the results in the folder: ./results_data_science/")
        else:
            print(f"The task {task_id} did not complete successfully. Return code:", result.returncode)
            return False

    except Exception as e:
        print("An error occurred while running the task:", str(e))
        return False

    finally:
        # Remove the temporary configuration file
        if os.path.exists(temp_config):
            os.remove(temp_config)

    return True


if __name__ == "__main__":
    print("\nSingle Data Science Task Runner")
    print("===================================")

    if not os.path.exists("run_data_science.py"):
        print("The file run_data_science.py was not found. Please run this script from the OSWorld main directory.")
        sys.exit(1)

    if not os.path.exists("evaluation_examples/examples/data_science/"):
        print("The data science example folder was not found. Please run setup first.")
        sys.exit(1)

    print("\nThis script will run a single data science task for testing.")
    print("Please make sure that:")
    print("  1. The OPENAI_API_KEY environment variable is set.")
    print("  2. VMware is properly configured.")
    print("  3. All dependencies are installed.")
    print()

    response = input("Do you want to continue? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled by the user.")
        sys.exit(0)

    if run_single_data_science_task():
        print("\nThe test finished successfully.")
    else:
        print("\nThe test did not complete successfully. Please check the logs for details.")
        sys.exit(1)
