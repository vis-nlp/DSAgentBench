#!/usr/bin/env python3
"""
Script to run data science task evaluations using GPT-4 and other models.
This extends OSWorld for the data science benchmark evaluation.
"""

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Dict, Any
from tqdm import tqdm

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from mm_agents.agent import PromptAgent
from mm_agents.openai_cua_agent import OpenAICUAAgent
from mm_agents.jedi_3b_agent import JediAgent3B
from mm_agents.jedi_7b_agent import JediAgent7B
from mm_agents.uitars_agent import UITARSAgent

# from mm_agents.gui_agents.s2.agents.agent_s import AgentS2
# from mm_agents.gui_agents.s2.agents.grounding import OSWorldACI

# from mm_agents.gat1_agent import GTA1Agent


# --------------------------- Logger Configuration ---------------------------
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", f"data_science-{datetime_str}.log"), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", f"data_science_debug-{datetime_str}.log"), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s"
)

for h in (file_handler, debug_handler, stdout_handler):
    h.setFormatter(formatter)
    logger.addHandler(h)

logger = logging.getLogger("desktopenv.data_science")
# ---------------------------------------------------------------------------


def config() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run data science task evaluations on the OSWorld benchmark."
    )

    # Environment configuration
    # parser.add_argument(
    #     "--path_to_vm",
    #     type=str,
    #     default=r"/home/ivlr/Study/Research/DSAgent/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx",
    #     help="Path to the VMware virtual machine to use for all tasks."
    # )

    # parser.add_argument(
    #     "--provider_name",
    #     type=str,
    #     default="vmware",
    #     help="Virtualization provider (vmware, docker, aws, azure, gcp, virtualbox)"
    # )

    # parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    parser.add_argument("--action_space", type=str, default="pyautogui")

    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot_a11y_tree",
        help="Observation type used by the environment"
    )

    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=10)

    # Agent configuration
    parser.add_argument("--max_trajectory_length", type=int, default=5)
    parser.add_argument("--test_config_base_dir", type=str, default="/home/ivlr/DSWorld/evaluation_examples")

    # LLM configuration
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--stop_token", type=str, default=None)

    # Example configuration
    parser.add_argument("--domain", type=str, default="data_science")
    parser.add_argument("--test_all_meta_path", type=str, default="/home/ivlr/DSWorld/evaluation_examples/test_small.json")

    # Logging and results
    parser.add_argument("--result_dir", type=str, default="/home/ivlr/DSWorld/results_data_science")

    # Task-specific
    parser.add_argument("--task_timeout", type=int, default=1800,
                        help="Maximum time per task in seconds")
    parser.add_argument("--enable_jupyter", action="store_true",
                        help="Enable Jupyter notebook support")
    parser.add_argument("--enable_vscode", action="store_true",
                        help="Enable Visual Studio Code support")

    return parser.parse_args()


def run_data_science_tasks(args: argparse.Namespace) -> None:
    """Run all data science tasks listed in the provided configuration file."""

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain == "data_science" and "data_science" in test_all_meta:
        test_meta = {"data_science": test_all_meta["data_science"]}
    elif args.domain in test_all_meta:
        test_meta = {args.domain: test_all_meta[args.domain]}
    else:
        test_meta = {}

    if not test_meta:
        logger.error(f"No tasks found for the selected domain: {args.domain}")
        return

    scores = []
    max_steps = args.max_steps
    logger.info("Configuration arguments: %s", args)


    # Initialize the agent
    agent = PromptAgent(
        model=args.model_name,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        max_trajectory_length=args.max_trajectory_length,
    )

    # env = DesktopEnv(
    #     provider_name=args.provider_name,
    #     path_to_vm=args.path_to_vm,
    #     action_space=agent.action_space,
    #     screen_size=(args.screen_width, args.screen_height),
    #     headless=args.headless,
    #     os_type="Ubuntu",
    #     require_a11y_tree=args.observation_type in
    #     ["a11y_tree", "screenshot_a11y_tree", "som"],
    # )
    env = DesktopEnv(
        provider_name="docker",
        os_type="Ubuntu",
        headless=True,
        action_space="pyautogui",
        screen_size=(1920, 1080),
    )

    # Create agent using the provided setup and model name
    if args.model_name in ['computer-use-preview']:
        agent = OpenAICUAAgent(
            env=env,
            model=args.model_name,
            action_space="pyautogui",
            observation_type=args.observation_type, #"screenshot_a11y_tree", #"som",
            max_trajectory_length=3,
        )
    elif args.model_name == "Jedi-3B-1080p":
        agent = JediAgent3B(
            executor_model=args.model_name,
            action_space="pyautogui",
            observation_type=args.observation_type, #"screenshot_a11y_tree", #"som",
            max_steps=25
        )
    elif args.model_name == "Jedi-7B-1080p":
        agent = JediAgent7B(
            executor_model=args.model_name,
            action_space="pyautogui",
            observation_type=args.observation_type, #"screenshot_a11y_tree", #"som",
            max_steps=25
        )
    elif args.model_name == "UI-TARS-1.5-7B":
        agent = UITARSAgent(
            model = args.model_name,
            action_space="pyautogui",
            observation_type=args.observation_type, #"screenshot_a11y_tree", #"som",
            max_trajectory_length=25 # This is treated as max steps by Ui Tars
        )
    # elif args.model_name in ['GAT1-7b']:
    #     agent = GTA1Agent(
    #         platform="ubuntu",
    #         action_space="pyautogui",
    #         observation_type=args.observation_type, #"screenshot_a11y_tree", #"som",
    #         max_steps=25 # This is treated as max steps by Ui Tars
    #     )
        
    # elif args.model_name in ['S2']:

    #     engine_params = {
    #         "engine_type": "gemini",
    #         "model":"gemini-2.5-pro",
    #         "base_url": os.environ["GEMINI_ENDPOINT_URL"],
    #         "api_key": os.environ["GEMINI_API_KEY"],
    #     }


    #     grounding_height = None
    #     grounding_model_resize_width = 1366
    #     screen_height = 1080
    #     screen_width = 1920
    #     # If not provided, use the aspect ratio of the screen to compute the height
    #     if grounding_height is None:
    #         grounding_height = (
    #             screen_height
    #             * grounding_model_resize_width
    #             / screen_width
    #         )

    #     engine_params_for_grounding = {
    #         "engine_type": "gemini",
    #         "model": "gemini-2.5-pro",
    #         "grounding_width": grounding_model_resize_width,
    #         "grounding_height": grounding_height,
    #     }

    #     # NEW!
    #     grounding_agent = OSWorldACI(
    #         platform="linux",
    #         planner_model=engine_params,
    #         engine_params_for_grounding=engine_params_for_grounding,
    #         width=screen_width,
    #         height=screen_height,
    #     )

    #     # NEW!
    #     # change observation type name.
    #     observation_for_agent = args.setup_name
    #     if observation_for_agent == 'screenshot_a11y_tree':
    #     observation_for_agent = 'mixed'
    #     agent = AgentS2(
    #         engine_params,
    #         grounding_agent,
    #         platform="linux",
    #         action_space="pyautogui",
    #         observation_type=observation_for_agent,
    #         search_engine=None, #"Perplexica",
    #         memory_root_path=os.getcwd(),
    #         memory_folder_name="kb_s2",
    #         kb_release_tag="v0.2.2",
    #         embedding_engine_type="openai",
    #     )
    else:
        agent = PromptAgent(
            model=args.model_name,
            action_space="pyautogui",
            observation_type=args.observation_type, #"screenshot_a11y_tree", #"som",
            max_trajectory_length=25,
        )
        
    
    # Initialize environment (uses default VM path automatically)
    

    for domain in tqdm(test_meta, desc="Domain"):
        for example_id in tqdm(test_meta[domain], desc="Data Science Task", leave=False):
            config_file = os.path.join(
                args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
            )

            if not os.path.exists(config_file):
                logger.error(f"The configuration file was not found: {config_file}")
                continue

            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)

            logger.info(f"Running domain: {domain}")
            logger.info(f"Task ID: {example_id}")
            logger.info(f"Instruction: {example['instruction']}")

            example_result_dir = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                args.model_name,
                domain,
                example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)

            try:
                logger.info(f"Resetting the environment for task: {example_id}")
                env.reset(task_config=example)
                logger.info("Environment reset completed successfully.")
            except Exception as e:
                logger.error(f"Could not reset the environment for {example_id}: {e}")
                continue

            task_success = False
            try:
                logger.info(f"Starting execution of task: {example_id}")
                lib_run_single.run_single_example(
                    agent, env, example, max_steps,
                    example["instruction"], args, example_result_dir, scores,
                )
                task_success = True
                logger.info(f"Task {example_id} completed successfully.")

            except Exception as e:
                logger.error(f"An exception occurred during task {example_id}: {e}")
                logger.error(f"Error details: {str(e)}")

                try:
                    if hasattr(env, 'controller') and env.controller is not None:
                        env.controller.end_recording(
                            os.path.join(example_result_dir, "recording.mp4")
                        )
                except Exception as cleanup_e:
                    logger.error(f"Error during cleanup: {cleanup_e}")

                error_info = {
                    "Error": f"Exception in {domain}/{example_id}: {str(e)}",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "task_id": example_id
                }

                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(json.dumps(error_info) + "\n")

                with open(os.path.join(example_result_dir, "result.txt"), "w") as f:
                    f.write("0.0\n")
                scores.append(0.0)

                import gc, time
                gc.collect()
                time.sleep(5)

            logger.info(f"Finished processing task {example_id}. Success: {task_success}")
            logger.info(f"Total tasks processed so far: {len(scores)}")

        env.close()

        if scores:
            avg_score = sum(scores) / len(scores)
            logger.info(f"Total data science tasks completed: {len(scores)}")
            logger.info(f"Average success rate: {avg_score:.2f}")

            summary = {
                "total_tasks": len(scores),
                "average_score": avg_score,
                "individual_scores": scores,
                "model": args.model_name,
                "timestamp": datetime_str,
            }

            summary_file = os.path.join(
                args.result_dir, f"data_science_summary_{datetime_str}.json"
            )
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Summary saved to {summary_file}")
        else:
            logger.warning("No tasks completed successfully.")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()
    os.makedirs(args.result_dir, exist_ok=True)

    config_path = os.path.join(args.result_dir, f"config_{datetime_str}.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Starting data science benchmark evaluation.")
    logger.info(f"Model in use: {args.model_name}")
    logger.info(f"Results will be stored in: {args.result_dir}")

    run_data_science_tasks(args)

    logger.info("Data science benchmark evaluation completed.")
